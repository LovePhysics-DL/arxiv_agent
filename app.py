import json
import os
import re

import arxiv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

app = Flask(__name__)
# 确保返回的 JSON 保持中文不转义
app.config.update(JSON_AS_ASCII=False)

APICORE_API_BASE = os.getenv('APICORE_API_BASE', 'https://api.apicore.ai/v1')
APICORE_API_KEY = os.getenv('APICORE_API_KEY', 'sk-I22ZX6GXD7gBGMj34kCEeXPHPrgbKgrxLlCzSTZlm8EfS0sM')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5')
STREAM_PER_QUERY_LIMIT = int(os.getenv('STREAM_PER_QUERY_LIMIT', '12'))
MAX_FILTER_CANDIDATES = int(os.getenv('MAX_FILTER_CANDIDATES', '12'))
MAX_REFLECTION_ATTEMPTS = int(os.getenv('MAX_REFLECTION_ATTEMPTS', '1'))

SORT_MAPPING = {
    'relevance': arxiv.SortCriterion.Relevance,
    'submitted_date': arxiv.SortCriterion.SubmittedDate,
    'last_updated': arxiv.SortCriterion.LastUpdatedDate,
}
MIN_RESULTS = 1
MAX_RESULTS = 100


def resolve_sort(sort_value: str):
    if sort_value not in SORT_MAPPING:
        raise ValueError('排序方式不支持')
    return SORT_MAPPING[sort_value]


def validate_result_count(raw_value) -> int:
    try:
        count = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError('结果数量必须是整数')
    if not (MIN_RESULTS <= count <= MAX_RESULTS):
        raise ValueError(f'结果数量需在 {MIN_RESULTS} 到 {MAX_RESULTS} 之间')
    return count


def perform_arxiv_search(query: str, sort_criterion, max_results: int, include_extra: bool = False):
    client = arxiv.Client(num_retries=2, delay_seconds=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    results = []
    for result in client.results(search):
        entry = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'published': result.published.strftime('%Y-%m-%d') if result.published else None,
            'arxiv_id': result.entry_id.split('/')[-1],
            'link': result.entry_id,
        }
        if include_extra:
            entry['pdf_url'] = getattr(result, 'pdf_url', None)
            entry['primary_category'] = getattr(result, 'primary_category', None)
        results.append(entry)
    return results


def extract_json_fragment(text: str) -> str:
    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if code_block:
        return code_block.group(1).strip()
    return text.strip()


def parse_ai_query(ai_output: str, fallback_query: str):
    cleaned = extract_json_fragment(ai_output)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        query = (payload.get('query') or '').strip()
        reasoning = (payload.get('reasoning') or '').strip()
        if query:
            return query, reasoning or ai_output.strip()

    match = re.search(r'query\s*[:：]\s*(.+)', ai_output, re.I)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted, ai_output.strip()

    return fallback_query, ai_output.strip()


def parse_multi_queries(ai_output: str, fallback_query: str, expected: int = 3):
    cleaned = extract_json_fragment(ai_output)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    queries = []
    if isinstance(payload, dict):
        raw_list = payload.get('queries') or payload.get('query_list')
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict):
                    candidate = (item.get('query') or item.get('text') or '').strip()
                else:
                    candidate = str(item).strip()
                if candidate:
                    queries.append(candidate)
    if not queries:
        inline = re.findall(r'"([^"]+)"', ai_output)
        queries.extend(inline)

    cleaned_set = []
    seen = set()
    for text in queries:
        if text and text not in seen:
            cleaned_set.append(text)
            seen.add(text)
        if len(cleaned_set) >= expected:
            break

    if not cleaned_set:
        cleaned_set = [fallback_query]

    while len(cleaned_set) < expected:
        cleaned_set.append(f"{fallback_query} variation {len(cleaned_set) + 1}")

    return cleaned_set[:expected]


def deduplicate_results(results):
    seen = set()
    deduped = []
    for item in results:
        identifier = item.get('arxiv_id')
        if identifier and identifier not in seen:
            seen.add(identifier)
            deduped.append(item)
    return deduped


QUERY_PROMPT = PromptTemplate(
    input_variables=["user_query"],
    template="""
你是一个专业的学术文献检索助手。用户会描述他们的研究需求，你需要：
1. 理解用户的需求
2. 生成合适的 arXiv 搜索关键词（用英文）

请以 JSON 格式返回，包含以下字段：
- query: arXiv 搜索关键词
- reasoning: 你的思考过程

用户查询: {user_query}
"""
)

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["user_query"],
    template="""
基于以下研究需求生成 3 个不同角度的 arXiv 检索式（英文），确保覆盖核心概念、相关方法和潜在交叉方向。
请严格输出 JSON，字段：
{{
  "queries": [
    {{"label": "核心方向", "query": "..."}},
    {{"label": "方法相关", "query": "..."}},
    {{"label": "交叉领域", "query": "..."}}
  ]
}}

用户需求：{user_query}
"""
)

FILTER_PROMPT = PromptTemplate(
    input_variables=["user_query", "limit", "candidates_json"],
    template="""
你收到一组候选论文（JSON 数组，含 arxiv_id、title、summary）。请结合用户需求，从中挑选最相关的 {limit} 篇以内论文。
输出 JSON：
{{
  "selected": [
    {{"arxiv_id": "...", "reason": "简短原因"}}
  ],
  "summary": "对整体检索结果的概述"
}}

用户需求：{user_query}
候选论文：{candidates_json}
"""
)


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=APICORE_API_BASE,
        openai_api_key=APICORE_API_KEY,
        temperature=0.1,
    )


REFLECTION_PROMPT = PromptTemplate(
    input_variables=["user_query", "previous_queries", "feedback"],
    template="""
你生成的检索式未检索到任何 arXiv 结果。请反思原因，并给出新的检索策略（英文）。
输出 JSON：
{{
  "analysis": "...",
  "queries": ["...", "...", "..."]
}}

用户需求：{user_query}
之前的检索式：{previous_queries}
系统反馈：{feedback}
"""
)


def parse_filter_output(ai_output: str):
    cleaned = extract_json_fragment(ai_output)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    selected = []
    summary = ''
    if isinstance(payload, dict):
        raw_selected = payload.get('selected')
        if isinstance(raw_selected, list):
            for item in raw_selected:
                if isinstance(item, dict):
                    arxiv_id = (item.get('arxiv_id') or item.get('id') or '').strip()
                    reason = (item.get('reason') or '').strip()
                    if arxiv_id:
                        selected.append({'arxiv_id': arxiv_id, 'reason': reason})
        summary = (payload.get('summary') or '').strip()
    return selected, summary


def refine_results_with_llm(llm, user_query: str, candidates: list, limit: int):
    if not candidates:
        return [], ''
    subset = candidates[: min(len(candidates), MAX_FILTER_CANDIDATES)]
    candidates_json = json.dumps(subset, ensure_ascii=False)
    chain = FILTER_PROMPT | llm
    response = chain.invoke({
        'user_query': user_query,
        'limit': limit,
        'candidates_json': candidates_json,
    })
    selected, summary = parse_filter_output(response.content)
    if not selected:
        return subset[:limit], summary
    mapping = {item['arxiv_id']: item for item in candidates}
    refined = []
    for pick in selected:
        paper = mapping.get(pick['arxiv_id'])
        if paper:
            paper_with_reason = dict(paper)
            if pick.get('reason'):
                paper_with_reason['reason'] = pick['reason']
            refined.append(paper_with_reason)
        if len(refined) >= limit:
            break
    if not refined:
        refined = subset[:limit]
    return refined, summary


def sse_event(event: str, payload: dict):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def reflect_queries(llm, user_query: str, previous_queries: list, feedback: str, expected: int = 3):
    try:
        chain = REFLECTION_PROMPT | llm
        response = chain.invoke({
            'user_query': user_query,
            'previous_queries': '\n'.join(previous_queries),
            'feedback': feedback,
        })
        content = response.content.strip()
    except Exception:
        return previous_queries[:expected], ''

    cleaned = extract_json_fragment(content)
    analysis = ''
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        analysis = (payload.get('analysis') or '').strip()
        raw_queries = payload.get('queries')
        if isinstance(raw_queries, list):
            queries = [str(item).strip() for item in raw_queries if str(item).strip()]
        else:
            queries = []
    else:
        queries = []

    if not queries:
        queries = parse_multi_queries(content, user_query, expected)

    if not queries:
        queries = previous_queries[:expected] or [user_query]

    return queries[:expected], analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': '查询关键词不能为空'}), 400
    
    sort_value = request.args.get('sort', 'relevance')
    raw_max = request.args.get('max_results', '20')
    try:
        sort_criterion = resolve_sort(sort_value)
        max_results = validate_result_count(raw_max)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        results = perform_arxiv_search(query, sort_criterion, max_results, include_extra=True)
        return jsonify(results)
    except Exception:
        # 不向前端泄露内部异常细节
        return jsonify({'error': '检索失败，请稍后再试'}), 502

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': '请求体必须是 JSON'}), 400

    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    sort_value = data.get('sort', 'relevance')
    raw_max = data.get('max_results', 10)
    try:
        sort_criterion = resolve_sort(sort_value)
        max_results = validate_result_count(raw_max)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        llm = get_llm()
        chain = QUERY_PROMPT | llm
        response = chain.invoke({"user_query": user_message})
        ai_output = response.content.strip()
        query, reasoning = parse_ai_query(ai_output, user_message)
        if not query:
            query = user_message
        if not reasoning:
            reasoning = 'LLM 输出为空，已使用原始输入'
    except Exception:
        return jsonify({'error': 'AI 处理失败，请检查 API 配置'}), 500

    attempted_queries = []
    reflection_notes = []
    results = []

    for attempt in range(MAX_REFLECTION_ATTEMPTS + 1):
        try:
            results = perform_arxiv_search(query, sort_criterion, max_results)
        except Exception:
            return jsonify({'error': '检索失败，请稍后再试'}), 502

        attempted_queries.append(query)
        if results:
            break

        if attempt >= MAX_REFLECTION_ATTEMPTS:
            break

        new_queries, analysis = reflect_queries(
            llm,
            user_message,
            attempted_queries,
            '未检索到文献，请重新审视关键词',
            expected=3,
        )
        if analysis:
            reflection_notes.append(analysis)

        # 选择一个尚未尝试的新检索式
        next_query = None
        for candidate in new_queries:
            if candidate not in attempted_queries:
                next_query = candidate
                break
        if not next_query and new_queries:
            next_query = new_queries[0]
        query = next_query or user_message

    combined_reasoning = reasoning
    if reflection_notes:
        combined_reasoning = f"{reasoning}\n反思: " + "；".join(reflection_notes)

    return jsonify({
        'reasoning': combined_reasoning,
        'query': query,
        'attempted_queries': attempted_queries,
        'sort': sort_value,
        'max_results': max_results,
        'results': results
    })


@app.route('/ai-search-stream', methods=['GET'])
def ai_search_stream():
    user_message = request.args.get('q', '').strip()
    if not user_message:
        return jsonify({'error': '查询关键词不能为空'}), 400

    sort_value = request.args.get('sort', 'relevance')
    raw_max = request.args.get('max_results', '20')
    try:
        sort_criterion = resolve_sort(sort_value)
        max_results = validate_result_count(raw_max)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    def generate_stream():
        try:
            llm = get_llm()
            yield sse_event('progress', {
                'percent': 5,
                'message': 'AI 正在理解查询意图…'
            })

            multi_chain = MULTI_QUERY_PROMPT | llm
            multi_resp = multi_chain.invoke({'user_query': user_message})
            ai_queries = parse_multi_queries(multi_resp.content.strip(), user_message)
            yield sse_event('progress', {
                'percent': 20,
                'message': '已生成多组检索式',
                'queries': ai_queries
            })

            per_query_limit = min(MAX_RESULTS, max(max_results, STREAM_PER_QUERY_LIMIT))
            all_queries = list(ai_queries)
            reflection_notes = []
            deduped = []

            for attempt in range(MAX_REFLECTION_ATTEMPTS + 1):
                aggregated = []
                current_queries = list(ai_queries)
                for idx, ai_query in enumerate(current_queries, start=1):
                    before_percent = 20 + int(30 * (idx - 1) / max(1, len(current_queries))) + attempt * 5
                    after_percent = 20 + int(30 * idx / max(1, len(current_queries))) + attempt * 5
                    yield sse_event('progress', {
                        'percent': min(before_percent, 90),
                        'message': f'正在执行检索 {idx}/{len(current_queries)} (第 {attempt + 1} 轮)',
                        'detail': ai_query
                    })
                    partial = perform_arxiv_search(ai_query, sort_criterion, per_query_limit, include_extra=True)
                    aggregated.extend(partial)
                    yield sse_event('progress', {
                        'percent': min(after_percent, 90),
                        'message': f'检索 {idx} 完成 (第 {attempt + 1} 轮)',
                        'detail': f'获取 {len(partial)} 篇文献'
                    })

                for q in current_queries:
                    if q not in all_queries:
                        all_queries.append(q)

                deduped = deduplicate_results(aggregated)
                if deduped or attempt >= MAX_REFLECTION_ATTEMPTS:
                    break

                yield sse_event('progress', {
                    'percent': 60 + attempt * 10,
                    'message': 'AI 正在反思检索策略…',
                    'detail': f'上一轮未检索到结果（尝试 {attempt + 1}）'
                })

                new_queries, analysis = reflect_queries(
                    llm,
                    user_message,
                    all_queries,
                    '当前检索式未返回结果，请调整关键词或扩展同义词',
                    expected=3,
                )
                if analysis:
                    reflection_notes.append(analysis)
                    yield sse_event('progress', {
                        'percent': 62 + attempt * 5,
                        'message': 'AI 反思完成',
                        'detail': analysis
                    })

                ai_queries = new_queries
                yield sse_event('progress', {
                    'percent': 65 + attempt * 5,
                    'message': 'AI 已生成新的检索式',
                    'queries': ai_queries
                })

            yield sse_event('progress', {
                'percent': 65,
                'message': 'AI 正在阅读摘要进行筛选…',
                'detail': f'候选 {len(deduped)} 篇'
            })

            refined, summary = refine_results_with_llm(llm, user_message, deduped, max_results)
            limited = refined[:max_results]
            yield sse_event('results', {
                'results': limited,
                'queries': all_queries,
                'filter_summary': summary,
                'reflection_notes': reflection_notes
            })
            yield sse_event('complete', {'status': 'done'})
        except Exception as exc:
            yield sse_event('failure', {'message': str(exc)})

    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

if __name__ == '__main__':
    # 通过环境变量控制调试开关（默认开启便于开发）
    debug_flag = os.getenv('FLASK_DEBUG', '1') in ('1', 'true', 'True')
    app.run(debug=debug_flag)
