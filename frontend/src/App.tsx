import {
  BugOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EditOutlined,
  EyeOutlined,
  FileTextOutlined,
  InfoCircleOutlined,
  MessageOutlined,
  PlusOutlined,
  ReloadOutlined,
  SendOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Collapse,
  Descriptions,
  Drawer,
  Empty,
  Form,
  Input,
  Layout,
  List,
  Modal,
  Segmented,
  Space,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
  message as toast,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, ArtifactItem, ChatMessage, ChatMode, ChatStreamEvent, Citation, DocumentItem, IndexTask, Session, TraceItem, TraceStats } from './api';

const { Header, Sider, Content } = Layout;
const { Text, Title, Paragraph } = Typography;

const documentColumns = (
  onOpen: (fileId: string) => void,
  onReindex: (fileId: string) => void,
  onDelete: (fileId: string) => void,
): ColumnsType<DocumentItem> => [
  {
    title: '文件',
    dataIndex: 'file_name',
    ellipsis: true,
    render: (value: string) => (
      <Space size={6}>
        <FileTextOutlined />
        <Text title={value}>{value}</Text>
      </Space>
    ),
  },
  {
    title: '分块',
    dataIndex: 'chunk_count',
    width: 72,
  },
  {
    title: '状态',
    dataIndex: 'status',
    width: 96,
    render: (value: string) => <StatusTag status={value} />,
  },
  {
    title: '操作',
    width: 148,
    render: (_, record) => (
      <Space size={4}>
        <Tooltip title="Open file">
          <Button size="small" icon={<EyeOutlined />} onClick={() => onOpen(record.id)} />
        </Tooltip>
        <Tooltip title="重建索引">
          <Button size="small" icon={<ReloadOutlined />} onClick={() => onReindex(record.id)} />
        </Tooltip>
        <Tooltip title="删除">
          <Button danger size="small" icon={<DeleteOutlined />} onClick={() => onDelete(record.id)} />
        </Tooltip>
      </Space>
    ),
  },
];

const taskColumns: ColumnsType<IndexTask> = [
  {
    title: '文件',
    dataIndex: 'file_name',
    ellipsis: true,
  },
  {
    title: '状态',
    dataIndex: 'status',
    width: 96,
    render: (value: string) => <StatusTag status={value} />,
  },
  {
    title: '消息',
    dataIndex: 'message',
    ellipsis: true,
  },
];

function StatusTag({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  const color =
    normalized === 'completed'
      ? 'green'
      : normalized === 'failed'
        ? 'red'
        : normalized === 'processing'
          ? 'blue'
          : 'gold';
  return <Tag color={color}>{status || 'unknown'}</Tag>;
}

function asArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : [];
}

function formatTraceValue(value: unknown): string {
  if (value === undefined || value === null || value === '') {
    return '';
  }
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function renderAnswerWithCitations(
  content: string,
  citations: Citation[] | undefined,
  onCitationClick: (citation: Citation) => void,
) {
  const citationMap = new Map((citations ?? []).map((citation) => [String(citation.id), citation]));
  if (!citationMap.size) {
    return content;
  }

  const nodes = [];
  const pattern = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(content)) !== null) {
    const [label, id] = match;
    if (match.index > lastIndex) {
      nodes.push(content.slice(lastIndex, match.index));
    }
    const citation = citationMap.get(id);
    if (citation) {
      nodes.push(
        <Button
          key={`${id}-${match.index}`}
          className="citation-link"
          type="link"
          size="small"
          onClick={() => onCitationClick(citation)}
        >
          {label}
        </Button>,
      );
    } else {
      nodes.push(label);
    }
    lastIndex = match.index + label.length;
  }
  if (lastIndex < content.length) {
    nodes.push(content.slice(lastIndex));
  }
  return nodes;
}

function TracePanel({
  traces,
  stats,
  onCitationClick,
}: {
  traces: TraceItem[];
  stats?: TraceStats;
  onCitationClick: (citation: Citation) => void;
}) {
  const items = traces.map((trace) => {
    const debug = trace.retrieval_debug as {
      timing_ms?: Record<string, unknown>;
      final?: unknown;
    };
    const finalDocs = asArray(debug.final);
    const traceCitations = trace.citations ?? [];
    return {
      key: String(trace.id),
      label: (
        <Space className="trace-label">
          <Tag color={trace.error ? 'red' : 'green'}>{trace.error ? 'ERROR' : 'OK'}</Tag>
          <Text>#{trace.id}</Text>
          <Text type="secondary">{trace.created_at}</Text>
        </Space>
      ),
      children: (
        <Space direction="vertical" size={12} className="trace-body">
          <div>
            <Text strong>用户输入</Text>
            <Paragraph className="trace-text">{trace.user_input}</Paragraph>
          </div>
          {trace.answer ? (
            <div>
              <Text strong>回答</Text>
              <Paragraph className="trace-text">
                {renderAnswerWithCitations(trace.answer, traceCitations, onCitationClick)}
              </Paragraph>
            </div>
          ) : null}
          {traceCitations.length ? (
            <div>
              <Text strong>引用原文</Text>
              <Space className="citation-list" size={[6, 6]} wrap>
                {traceCitations.map((citation) => (
                  <Button key={citation.id} size="small" onClick={() => onCitationClick(citation)}>
                    [{citation.id}] {citation.source}
                  </Button>
                ))}
              </Space>
            </div>
          ) : null}
          {trace.error ? <Alert type="error" showIcon message={trace.error} /> : null}
          {trace.tool_calls.length ? (
            <div>
              <Text strong>调用链路</Text>
              <Table
                size="small"
                rowKey={(_, index) => `${trace.id}-tool-${index}`}
                pagination={false}
                dataSource={trace.tool_calls}
                columns={[
                  {
                    title: '工具',
                    dataIndex: 'tool',
                    width: 132,
                    ellipsis: true,
                    render: (value: unknown, record) => (
                      <Text title={formatTraceValue(value)}>{formatTraceValue(record.display_name) || formatTraceValue(value)}</Text>
                    ),
                  },
                  {
                    title: '状态',
                    dataIndex: 'status',
                    width: 96,
                    render: (value: string) => <StatusTag status={value || 'completed'} />,
                  },
                  {
                    title: '输入',
                    dataIndex: 'input',
                    ellipsis: true,
                    render: (value: unknown) => <Text title={formatTraceValue(value)}>{formatTraceValue(value)}</Text>,
                  },
                  {
                    title: '输出预览',
                    dataIndex: 'output_preview',
                    ellipsis: true,
                    render: (value: unknown) => <Text title={formatTraceValue(value)}>{formatTraceValue(value)}</Text>,
                  },
                ]}
              />
            </div>
          ) : null}
          {debug.timing_ms ? (
            <div>
              <Text strong>耗时</Text>
              <pre className="json-block">{JSON.stringify(debug.timing_ms, null, 2)}</pre>
            </div>
          ) : null}
          {finalDocs.length ? (
            <Table
              size="small"
              rowKey={(record) => String(record.chunk_id ?? record.rank ?? Math.random())}
              pagination={false}
              dataSource={finalDocs}
              columns={[
                { title: 'Rank', dataIndex: 'rank', width: 72 },
                { title: 'Score', dataIndex: 'score', width: 96 },
                { title: 'Source', dataIndex: 'source', ellipsis: true },
                { title: 'Preview', dataIndex: 'preview', ellipsis: true },
              ]}
            />
          ) : null}
        </Space>
      ),
    };
  });

  return (
    <div className="side-section">
      <Space className="section-title">
        <BugOutlined />
        <Text strong>Trace</Text>
        {stats ? <Text type="secondary">{stats.successes}/{stats.total}</Text> : null}
      </Space>
      {items.length ? <Collapse size="small" items={items} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />}
    </div>
  );
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactItem[]>([]);
  const [tasks, setTasks] = useState<IndexTask[]>([]);
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [stats, setStats] = useState<TraceStats>();
  const [input, setInput] = useState('');
  const [loadingApp, setLoadingApp] = useState(true);
  const [loadingSession, setLoadingSession] = useState(false);
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [apiReady, setApiReady] = useState(true);
  const [chatMode, setChatMode] = useState<ChatMode>('chat');
  const [workspaceAvailable, setWorkspaceAvailable] = useState(false);
  const [workspaceOutput, setWorkspaceOutput] = useState<string[]>([]);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameTitle, setRenameTitle] = useState('');
  const [renamingSession, setRenamingSession] = useState<Session>();
  const [selectedCitation, setSelectedCitation] = useState<Citation>();
  const messageListRef = useRef<HTMLDivElement | null>(null);

  const currentSession = useMemo(
    () => sessions.find((session) => session.id === currentSessionId),
    [currentSessionId, sessions],
  );

  const loadSessions = useCallback(async () => {
    const response = await api.sessions();
    setSessions(response.sessions);
    return response.sessions;
  }, []);

  const loadSessionData = useCallback(async (sessionId: string) => {
    setLoadingSession(true);
    try {
      const [messageRes, docRes, artifactRes, taskRes, traceRes, statRes] = await Promise.all([
        api.messages(sessionId),
        api.documents(sessionId),
        api.artifacts(sessionId),
        api.indexTasks(sessionId),
        api.traces(sessionId),
        api.traceStats(sessionId),
      ]);
      setMessages(messageRes.messages);
      setDocuments(docRes.documents);
      setArtifacts(artifactRes.artifacts);
      setTasks(taskRes.tasks);
      setTraces(traceRes.traces);
      setStats(statRes.stats);
    } finally {
      setLoadingSession(false);
    }
  }, []);

  const loadSessionSideData = useCallback(async (sessionId: string) => {
    const [docRes, artifactRes, taskRes, traceRes, statRes] = await Promise.all([
      api.documents(sessionId),
      api.artifacts(sessionId),
      api.indexTasks(sessionId),
      api.traces(sessionId),
      api.traceStats(sessionId),
    ]);
    setDocuments(docRes.documents);
    setArtifacts(artifactRes.artifacts);
    setTasks(taskRes.tasks);
    setTraces(traceRes.traces);
    setStats(statRes.stats);
  }, []);

  const bootstrap = useCallback(async () => {
    setLoadingApp(true);
    try {
      const health = await api.health();
      setWorkspaceAvailable(Boolean(health.workspace?.available));
      setApiReady(true);
      let loadedSessions = await loadSessions();
      if (!loadedSessions.length) {
        const created = await api.createSession();
        loadedSessions = [created];
        setSessions(loadedSessions);
      }
      setCurrentSessionId(loadedSessions[0].id);
      await loadSessionData(loadedSessions[0].id);
    } catch (error) {
      setApiReady(false);
      toast.error(error instanceof Error ? error.message : '后端服务不可用');
    } finally {
      setLoadingApp(false);
    }
  }, [loadSessionData, loadSessions]);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  useEffect(() => {
    if (!currentSessionId || !apiReady) {
      return;
    }
    const timer = window.setInterval(() => {
      void api
        .indexTasks(currentSessionId)
        .then((response) => setTasks(response.tasks))
        .catch(() => setApiReady(false));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [apiReady, currentSessionId]);

  const scrollMessagesToBottom = useCallback(() => {
    window.requestAnimationFrame(() => {
      const messageList = messageListRef.current;
      if (messageList) {
        messageList.scrollTop = messageList.scrollHeight;
      }
    });
  }, []);

  useEffect(() => {
    scrollMessagesToBottom();
  }, [messages, scrollMessagesToBottom]);

  const refreshCurrent = useCallback(async () => {
    if (!currentSessionId) {
      return;
    }
    try {
      await Promise.all([loadSessions(), loadSessionData(currentSessionId)]);
      setApiReady(true);
    } catch (error) {
      setApiReady(false);
      throw error;
    }
  }, [currentSessionId, loadSessionData, loadSessions]);

  const refreshCurrentSideData = useCallback(async () => {
    if (!currentSessionId) {
      return;
    }
    await Promise.all([loadSessions(), loadSessionSideData(currentSessionId)]);
  }, [currentSessionId, loadSessionSideData, loadSessions]);

  const handleCreateSession = async () => {
    try {
      const created = await api.createSession();
      const loadedSessions = await loadSessions();
      setCurrentSessionId(created.id);
      setSessions(loadedSessions);
      await loadSessionData(created.id);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '新建会话失败');
    }
  };

  const handleSelectSession = async (sessionId: string) => {
    setCurrentSessionId(sessionId);
    await loadSessionData(sessionId);
  };

  const handleDeleteSession = (sessionId: string) => {
    Modal.confirm({
      title: '删除会话',
      content: '该会话的元数据和文档记录会被删除。',
      okButtonProps: { danger: true },
      onOk: async () => {
        await api.deleteSession(sessionId);
        let loadedSessions = await loadSessions();
        if (!loadedSessions.length) {
          const created = await api.createSession();
          loadedSessions = [created];
          setSessions(loadedSessions);
        }
        const nextSession = loadedSessions[0];
        setCurrentSessionId(nextSession.id);
        await loadSessionData(nextSession.id);
      },
    });
  };

  const openRename = (session: Session) => {
    setRenamingSession(session);
    setRenameTitle(session.title);
    setRenameOpen(true);
  };

  const handleRename = async () => {
    if (!renamingSession || !renameTitle.trim()) {
      return;
    }
    await api.renameSession(renamingSession.id, renameTitle.trim());
    await loadSessions();
    setRenameOpen(false);
    setRenamingSession(undefined);
  };

  const updateLastAssistant = useCallback((updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((previous) => {
      const next = [...previous];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'assistant') {
          next[index] = updater(next[index]);
          break;
        }
      }
      return next;
    });
    scrollMessagesToBottom();
  }, [scrollMessagesToBottom]);

  const handleStreamEvent = useCallback(
    (event: ChatStreamEvent) => {
      if (event.type === 'token' && event.token) {
        updateLastAssistant((message) => ({
          ...message,
          content: `${message.content}${event.token}`,
          status: message.status?.startsWith('正在调用') ? message.status : '正在生成回答',
        }));
      } else if (event.type === 'tool_start') {
        updateLastAssistant((message) => ({ ...message, status: event.message || `正在调用 ${event.tool}` }));
      } else if (event.type === 'tool_end') {
        updateLastAssistant((message) => ({ ...message, status: event.message || `${event.tool} 调用完成` }));
      } else if (event.type === 'sandbox_start' || event.type === 'sandbox_ready' || event.type === 'sandbox_error') {
        setWorkspaceOutput((previous) => [...previous, event.message || event.error || event.type]);
      } else if (event.type === 'process_output' && event.output_preview) {
        setWorkspaceOutput((previous) => [...previous, event.output_preview || '']);
      } else if (event.type === 'status') {
        updateLastAssistant((message) => ({ ...message, status: event.message || '正在思考' }));
      } else if (event.type === 'done') {
        updateLastAssistant((message) => ({
          ...message,
          content: event.answer || message.content,
          citations: event.citations ?? [],
          status: undefined,
        }));
      } else if (event.type === 'error') {
        updateLastAssistant((message) => ({ ...message, status: event.error || '发送失败' }));
      }
    },
    [updateLastAssistant],
  );

  const handleSend = async () => {
    const text = input.trim();
    if (!currentSessionId || !text || sending) {
      return;
    }
    setInput('');
    setSending(true);
    if (chatMode === 'workspace') {
      setWorkspaceOutput([]);
    }
    setMessages((previous) => [
      ...previous,
      { role: 'user', content: text },
      { role: 'assistant', content: '', status: '正在思考' },
    ]);
    try {
      await api.chatStream(currentSessionId, text, handleStreamEvent, chatMode);
      await refreshCurrentSideData();
    } catch (error) {
      updateLastAssistant((message) => ({
        ...message,
        status: error instanceof Error ? error.message : '发送失败',
      }));
      toast.error(error instanceof Error ? error.message : '发送失败');
    } finally {
      setSending(false);
    }
  };

  const handleUpload = async (file: File) => {
    if (!currentSessionId) {
      return false;
    }
    setUploading(true);
    try {
      const response = await api.uploadDocument(currentSessionId, file);
      toast.success(response.message);
      await refreshCurrent();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '上传失败');
    } finally {
      setUploading(false);
    }
    return false;
  };

  const handleReindex = async (fileId: string) => {
    if (!currentSessionId) {
      return;
    }
    try {
      const response = await api.reindexDocument(currentSessionId, fileId);
      toast.success(response.message);
      await refreshCurrent();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '重建索引失败');
    }
  };

  const handleOpenDocument = (fileId: string) => {
    if (!currentSessionId) {
      return;
    }
    window.open(api.documentFileUrl(currentSessionId, fileId), '_blank', 'noopener,noreferrer');
  };

  const handleDeleteDocument = (fileId: string) => {
    if (!currentSessionId) {
      return;
    }
    Modal.confirm({
      title: '删除文档',
      content: '该文档记录和对应向量分块会被删除。',
      okButtonProps: { danger: true },
      onOk: async () => {
        const response = await api.deleteDocument(currentSessionId, fileId);
        toast.success(response.message);
        await refreshCurrent();
      },
    });
  };

  const openCitation = useCallback((citation: Citation) => {
    setSelectedCitation(citation);
  }, []);

  return (
    <Layout className="app-shell">
      <Sider width={280} theme="light" className="session-sider">
        <div className="brand">
          <Title level={4}>Agentic RAG</Title>
        </div>
        <Button type="primary" icon={<PlusOutlined />} block onClick={handleCreateSession}>
          新建会话
        </Button>
        <List
          className="session-list"
          dataSource={sessions}
          locale={{ emptyText: '暂无会话' }}
          renderItem={(session) => (
            <List.Item
              className={session.id === currentSessionId ? 'session-item active' : 'session-item'}
              onClick={() => void handleSelectSession(session.id)}
              actions={[
                <Tooltip title="重命名" key="edit">
                  <Button
                    size="small"
                    type="text"
                    icon={<EditOutlined />}
                    onClick={(event) => {
                      event.stopPropagation();
                      openRename(session);
                    }}
                  />
                </Tooltip>,
                <Tooltip title="删除" key="delete">
                  <Button
                    size="small"
                    danger
                    type="text"
                    icon={<DeleteOutlined />}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDeleteSession(session.id);
                    }}
                  />
                </Tooltip>,
              ]}
            >
              <List.Item.Meta
                avatar={<MessageOutlined />}
                title={<Text ellipsis>{session.title || '未命名会话'}</Text>}
                description={<Text type="secondary">{session.id.slice(0, 8)}</Text>}
              />
            </List.Item>
          )}
        />
      </Sider>

      <Layout>
        <Header className="topbar">
          <div className="topbar-title">
            <Title level={4}>{currentSession?.title || '未命名会话'}</Title>
          </div>
          <Space>
            {!apiReady ? <Tag color="red">API 离线</Tag> : <Tag color="green">API 在线</Tag>}
            <Button icon={<ReloadOutlined />} onClick={() => void refreshCurrent()}>
              刷新
            </Button>
          </Space>
        </Header>

        <Content className="workspace">
          <section className="chat-panel">
            {loadingApp || loadingSession ? (
              <div className="center-state">
                <Spin />
              </div>
            ) : messages.length ? (
              <div className="message-list" ref={messageListRef}>
                {messages.map((item, index) => (
                  <div key={`${item.role}-${index}`} className={`message-row ${item.role === 'user' ? 'user' : 'assistant'}`}>
                    <div className="message-bubble">
                      <Text strong>{item.role === 'user' ? '用户' : '助手'}</Text>
                      {item.status ? (
                        <Space className="message-status" size={6}>
                          {item.role === 'assistant' && sending ? <Spin size="small" /> : null}
                          <Text type="secondary">{item.status}</Text>
                        </Space>
                      ) : null}
                      <Paragraph className="message-content">
                        {renderAnswerWithCitations(item.content, item.citations, openCitation)}
                      </Paragraph>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="center-state">
                <Empty description="暂无消息" />
              </div>
            )}

            {chatMode === 'workspace' && workspaceOutput.length ? (
              <div className="workspace-console" aria-live="polite">
                <Text strong>隔离工作区输出</Text>
                <pre>{workspaceOutput.join('\n')}</pre>
              </div>
            ) : null}

            <div className="composer">
              <div className="composer-mode">
                <Segmented
                  value={chatMode}
                  options={[
                    { label: '对话', value: 'chat' },
                    { label: '工作区', value: 'workspace', disabled: !workspaceAvailable },
                  ]}
                  onChange={(value) => setChatMode(value as ChatMode)}
                />
                <Text type="secondary">
                  {chatMode === 'workspace'
                    ? '命令和文件操作只在会话沙箱的 /workspace 内执行'
                    : workspaceAvailable
                      ? '当前保持原有对话能力'
                      : '工作区运行时尚未启用'}
                </Text>
              </div>
              <Input.TextArea
                value={input}
                disabled={sending || !currentSessionId}
                autoSize={{ minRows: 2, maxRows: 5 }}
                placeholder="请输入问题"
                onChange={(event) => setInput(event.target.value)}
                onPressEnter={(event) => {
                  if (!event.shiftKey) {
                    event.preventDefault();
                    void handleSend();
                  }
                }}
              />
              <Button type="primary" icon={<SendOutlined />} loading={sending} onClick={() => void handleSend()}>
                发送
              </Button>
            </div>
          </section>

          <aside className="right-panel">
            <div className="side-section">
              <Space className="section-title">
                <FileTextOutlined />
                <Text strong>文档</Text>
              </Space>
              <Upload
                accept=".txt,.pdf,.md,.markdown,.doc,.docx,.csv,.xlsx,.parquet"
                showUploadList={false}
                beforeUpload={(file) => {
                  void handleUpload(file);
                  return false;
                }}
              >
                <Button icon={<UploadOutlined />} loading={uploading} block>
                  上传文件
                </Button>
              </Upload>
              <Table
                size="small"
                rowKey="id"
                className="compact-table"
                columns={documentColumns(handleOpenDocument, handleReindex, handleDeleteDocument)}
                dataSource={documents}
                pagination={false}
                locale={{ emptyText: '暂无文档' }}
              />
            </div>

            <div className="side-section">
              <Space className="section-title">
                <DownloadOutlined />
                <Text strong>已保存产物</Text>
              </Space>
              <List
                size="small"
                dataSource={artifacts}
                locale={{ emptyText: '暂无已保存产物' }}
                renderItem={(artifact) => (
                  <List.Item
                    actions={[
                      <Button
                        key="download"
                        size="small"
                        icon={<DownloadOutlined />}
                        onClick={() => window.open(api.artifactFileUrl(currentSessionId || '', artifact.id), '_blank', 'noopener,noreferrer')}
                      />,
                    ]}
                  >
                    <List.Item.Meta title={artifact.name} description={`${artifact.size_bytes} bytes`} />
                  </List.Item>
                )}
              />
            </div>

            <div className="side-section">
              <Space className="section-title">
                <ReloadOutlined />
                <Space size={6}>
                  <Text strong>入库任务</Text>
                  <Tooltip title="大文件上传后会在后台解析、切分、向量化并写入本地知识库；这里显示这些后台任务的进度。">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              </Space>
              <Table
                size="small"
                rowKey="task_id"
                className="compact-table"
                columns={taskColumns}
                dataSource={tasks}
                pagination={false}
                locale={{ emptyText: '暂无任务' }}
              />
            </div>

            <TracePanel traces={traces} stats={stats} onCitationClick={openCitation} />
          </aside>
        </Content>
      </Layout>

      <Drawer
        title={selectedCitation ? `引用 [${selectedCitation.id}]` : '引用'}
        open={Boolean(selectedCitation)}
        width={520}
        onClose={() => setSelectedCitation(undefined)}
      >
        {selectedCitation ? (
          <Space direction="vertical" size={16} className="citation-drawer">
            <Descriptions size="small" column={1} bordered>
              <Descriptions.Item label="来源">{selectedCitation.source}</Descriptions.Item>
              <Descriptions.Item label="页码">{selectedCitation.page ?? '无'}</Descriptions.Item>
              <Descriptions.Item label="Chunk ID">{selectedCitation.chunk_id ?? '无'}</Descriptions.Item>
              <Descriptions.Item label="Score">{selectedCitation.score ?? '无'}</Descriptions.Item>
            </Descriptions>
            <div>
              <Text strong>原文内容</Text>
              <pre className="citation-content">{selectedCitation.content}</pre>
            </div>
          </Space>
        ) : null}
      </Drawer>

      <Modal
        title="重命名会话"
        open={renameOpen}
        onOk={() => void handleRename()}
        onCancel={() => setRenameOpen(false)}
        destroyOnClose
      >
        <Form layout="vertical">
          <Form.Item label="标题">
            <Input value={renameTitle} maxLength={80} onChange={(event) => setRenameTitle(event.target.value)} />
          </Form.Item>
        </Form>
      </Modal>
    </Layout>
  );
}
