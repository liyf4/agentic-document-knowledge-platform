export type Session = {
  id: string;
  title: string;
};

export type ChatMessage = {
  role: string;
  content: string;
  status?: string;
  citations?: Citation[];
};

export type ContextUsage = {
  model: string;
  context_window_tokens: number;
  used_tokens: number;
  usage_percent: number;
  prompt_tokens: number;
  completion_tokens: number;
  measurement: 'actual' | 'estimated';
  compacted: boolean;
  target_unreachable?: boolean;
  measured_at?: string | null;
  last_compacted_at?: string | null;
};

export type ChatMode = 'chat' | 'workspace';

export type WorkspaceCapabilities = {
  enabled: boolean;
  provider: string;
  sdk_available: boolean;
  available: boolean;
  network_enabled: boolean;
  workspace_root: string;
};

export type Citation = {
  id: string;
  source: string;
  page?: string | number | null;
  chunk_id?: string | number | null;
  score?: number | string | null;
  content: string;
  preview: string;
};

export type DocumentItem = {
  id: string;
  file_name: string;
  chunk_count: number;
  status: string;
  uploaded_at: string;
  mime_type?: string;
  category?: string;
  size_bytes?: number;
  parse_status?: string;
  ocr_status?: string;
  index_status?: string;
  scan_status?: string;
  table_profile?: Record<string, unknown>;
};

export type ArtifactItem = {
  id: string;
  name: string;
  size_bytes: number;
  mime_type: string;
  created_at: string;
  generated_by_command: string;
  retention_policy: string;
};

export type IndexTask = {
  task_id: string;
  session_id: string;
  file_id: string;
  file_name: string;
  status: string;
  message: string;
  error: string;
  created_at: string;
  updated_at: string;
};

export type TraceItem = {
  id: number;
  session_id: string;
  user_input: string;
  answer: string;
  retrieval_debug: Record<string, unknown>;
  tool_calls: Array<Record<string, unknown>>;
  citations: Citation[];
  error: string;
  created_at: string;
};

export type TraceStats = {
  total: number;
  failures: number;
  successes: number;
  last_trace_at: string;
};

export type ChatResponse = {
  answer: string;
  tool_calls: Array<Record<string, unknown>>;
  retrieval_debug: Record<string, unknown>;
  citations: Citation[];
  context_usage?: ContextUsage;
};

export type ChatStreamEvent = {
  type: string;
  token?: string;
  tool?: string;
  display_name?: string;
  input?: unknown;
  message?: string;
  answer?: string;
  tool_calls?: Array<Record<string, unknown>>;
  retrieval_debug?: Record<string, unknown>;
  citations?: Citation[];
  error?: string;
  output_preview?: string;
  available?: boolean;
  runtime?: Record<string, unknown> | null;
  context_usage?: ContextUsage;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(init?.headers ?? {}),
    },
  });

  const contentType = response.headers.get('content-type') ?? '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === 'object' && payload && 'detail' in payload ? String(payload.detail) : String(payload);
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return payload as T;
}

async function chatStream(
  sessionId: string,
  message: string,
  onEvent: (event: ChatStreamEvent) => void,
  mode: ChatMode = 'chat',
): Promise<ChatResponse> {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message, mode }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => undefined);
    const detail = payload && typeof payload === 'object' && 'detail' in payload ? String(payload.detail) : response.statusText;
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  if (!response.body) {
    throw new Error('Streaming response is not available in this browser.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResponse: ChatResponse = { answer: '', tool_calls: [], retrieval_debug: {}, citations: [] };
  let streamError = '';

  const parseBlock = (block: string) => {
    const lines = block.split(/\r?\n/);
    let type = 'message';
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith('event:')) {
        type = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) {
      return;
    }
    const payload = JSON.parse(dataLines.join('\n')) as Omit<ChatStreamEvent, 'type'>;
    const event = { type, ...payload };
    onEvent(event);
    if (type === 'done') {
      finalResponse = {
        answer: event.answer ?? '',
        tool_calls: event.tool_calls ?? [],
        retrieval_debug: event.retrieval_debug ?? {},
        citations: event.citations ?? [],
        context_usage: event.context_usage,
      };
    }
    if (type === 'error') {
      streamError = event.error || '发送失败';
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\n\n/);
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      parseBlock(block);
    }
    if (done) {
      break;
    }
  }
  if (buffer.trim()) {
    parseBlock(buffer);
  }
  if (streamError) {
    throw new Error(streamError);
  }
  return finalResponse;
}

export const api = {
  health: () => request<{ ok: boolean; zai_api_key_configured: boolean; workspace: WorkspaceCapabilities }>('/api/health'),
  sessions: () => request<{ sessions: Session[] }>('/api/sessions'),
  createSession: () => request<Session>('/api/sessions', { method: 'POST' }),
  renameSession: (sessionId: string, title: string) =>
    request<{ ok: boolean }>(`/api/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),
  deleteSession: (sessionId: string) => request<{ ok: boolean }>(`/api/sessions/${sessionId}`, { method: 'DELETE' }),
  messages: (sessionId: string) => request<{ messages: ChatMessage[] }>(`/api/sessions/${sessionId}/messages`),
  contextUsage: (sessionId: string) =>
    request<{ context_usage: ContextUsage }>(`/api/sessions/${sessionId}/context-usage`),
  chat: (sessionId: string, message: string, mode: ChatMode = 'chat') =>
    request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, message, mode }),
    }),
  chatStream,
  sandboxStatus: (sessionId: string) =>
    request<Record<string, unknown>>(`/api/sessions/${sessionId}/sandbox`),
  destroySandbox: (sessionId: string) =>
    request<{ ok: boolean; destroyed: boolean }>(`/api/sessions/${sessionId}/sandbox`, { method: 'DELETE' }),
  documents: (sessionId: string) => request<{ documents: DocumentItem[] }>(`/api/sessions/${sessionId}/documents`),
  uploadDocument: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<{ ok: boolean; mode: string; task_id?: string; message: string }>(`/api/sessions/${sessionId}/documents`, {
      method: 'POST',
      body: form,
    });
  },
  deleteDocument: (sessionId: string, fileId: string) =>
    request<{ ok: boolean; message: string }>(`/api/sessions/${sessionId}/documents/${fileId}`, { method: 'DELETE' }),
  reindexDocument: (sessionId: string, fileId: string) =>
    request<{ ok: boolean; message: string }>(`/api/sessions/${sessionId}/documents/${fileId}/reindex`, { method: 'POST' }),
  documentFileUrl: (sessionId: string, fileId: string) =>
    `/api/sessions/${encodeURIComponent(sessionId)}/documents/${encodeURIComponent(fileId)}/file`,
  artifacts: (sessionId: string) => request<{ artifacts: ArtifactItem[] }>(`/api/sessions/${sessionId}/artifacts`),
  artifactFileUrl: (sessionId: string, artifactId: string) =>
    `/api/sessions/${encodeURIComponent(sessionId)}/artifacts/${encodeURIComponent(artifactId)}/file`,
  indexTasks: (sessionId: string) => request<{ tasks: IndexTask[] }>(`/api/sessions/${sessionId}/index-tasks`),
  traces: (sessionId: string) => request<{ traces: TraceItem[] }>(`/api/sessions/${sessionId}/traces`),
  traceStats: (sessionId: string) => request<{ stats: TraceStats }>(`/api/sessions/${sessionId}/trace-stats`),
};
