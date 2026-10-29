import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy, Plus, Send, Square, Trash2, MessageSquare, X } from "lucide-react";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  loadChats,
  saveChats,
  type Chat,
  type Msg,
  type PendingApproval,
  type ToolResult,
} from "@/lib/chat-store";
import { getDemoUserId } from "@/lib/demo-user";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Parley — AI Chat" },
      {
        name: "description",
        content: "Chat with multiple AI models. Conversations stay in your browser.",
      },
      { property: "og:title", content: "Parley — AI Chat" },
      {
        property: "og:description",
        content: "Chat with multiple AI models. Conversations stay in your browser.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ChatApp,
});

// API base URL comes from configuration (VITE_API_BASE_URL); the API key lives in your backend.
const API_BASE = (
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined) || "http://localhost:8000"
).replace(/\/$/, "");

function ChatApp() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [input, setInput] = useState("");
  const [temperature, setTemperature] = useState("1");
  const [maxTokens, setMaxTokens] = useState("");
  const [extraParameters, setExtraParameters] = useState<
    { id: string; name: string; value: string }[]
  >([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([]);
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [approvalError, setApprovalError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const activeApproval = pendingApprovals[0] ?? null;

  const refreshModels = async () => {
    try {
      setError(null);
      const res = await fetch(`${API_BASE}/v1/models`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const j = await res.json();
      const ids: string[] = (j.data ?? j.models ?? j ?? [])
        .map((m: { id?: string; name?: string } | string) =>
          typeof m === "string" ? m : (m.id ?? m.name ?? ""),
        )
        .filter(Boolean);
      setModels(ids);
      const saved = localStorage.getItem("model");
      setModel(saved && ids.includes(saved) ? saved : (ids[0] ?? ""));
    } catch (e) {
      setModels([]);
      setError(
        `Couldn't load models from ${API_BASE} (${(e as Error).message}). Is your API running with CORS enabled?`,
      );
    }
  };

  useEffect(() => {
    const c = loadChats();
    setChats(c);
    setActiveId(c[0]?.id ?? null);
    setPendingApprovals(
      c.flatMap((chat) =>
        chat.messages.flatMap((message) =>
          (message.toolResults ?? [])
            .map((toolResult) => toPendingApproval(toolResult, chat.id))
            .filter((approval): approval is PendingApproval => approval !== null),
        ),
      ),
    );
    localStorage.removeItem("apiKey");
    localStorage.removeItem("baseUrl");
    refreshModels();
  }, []);

  const active = chats.find((c) => c.id === activeId) ?? null;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages]);

  const update = (fn: (c: Chat[]) => Chat[]) =>
    setChats((prev) => {
      const next = fn(prev);
      saveChats(next);
      return next;
    });

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    if (!model) {
      setError("Pick a model first.");
      return;
    }
    const parsedTemperature = Number(temperature);
    if (
      !temperature.trim() ||
      !Number.isFinite(parsedTemperature) ||
      parsedTemperature < 0 ||
      parsedTemperature > 2
    ) {
      setError("Temperature must be between 0 and 2.");
      return;
    }
    const parsedMaxTokens = maxTokens.trim() ? Number(maxTokens) : undefined;
    if (
      parsedMaxTokens !== undefined &&
      (!Number.isInteger(parsedMaxTokens) || parsedMaxTokens < 1 || parsedMaxTokens > 32768)
    ) {
      setError("Max tokens must be a whole number between 1 and 32768.");
      return;
    }
    const parameters: Record<string, unknown> = {
      temperature: parsedTemperature,
      ...(parsedMaxTokens === undefined ? {} : { max_tokens: parsedMaxTokens }),
    };
    const parameterNames = new Set<string>();
    const restrictedNames = new Set([
      "model",
      "messages",
      "stream",
      "api_key",
      "base_url",
      "default_headers",
      "extra_body",
      "extra_headers",
      "extra_query",
      "http_client",
      "max_retries",
      "organization",
      "project",
      "timeout",
      "temperature",
      "max_tokens",
    ]);
    try {
      for (const parameter of extraParameters) {
        const name = parameter.name.trim();
        if (!name && !parameter.value.trim()) continue;
        if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
          throw new Error(
            "Parameter names must use letters, numbers, and underscores, and start with a letter or underscore.",
          );
        }
        if (restrictedNames.has(name)) {
          throw new Error(
            `"${name}" is controlled by the app and cannot be added as an extra parameter.`,
          );
        }
        if (name in parameters || parameterNames.has(name)) {
          throw new Error(`Parameter "${name}" was added more than once.`);
        }
        if (!parameter.value.trim()) {
          throw new Error(`Enter a JSON value for "${name}".`);
        }
        parameters[name] = JSON.parse(parameter.value);
        parameterNames.add(name);
      }
    } catch (e) {
      setError((e as Error).message);
      return;
    }
    setError(null);
    setInput("");
    let id = activeId;
    let history: Msg[] = active?.messages ?? [];
    if (!active) {
      id = crypto.randomUUID();
      update((c) => [
        { id: id!, title: text.slice(0, 40), messages: [], updatedAt: Date.now() },
        ...c,
      ]);
      setActiveId(id);
      history = [];
    }
    const msgs: Msg[] = [...history, { role: "user", content: text }];
    const setMsgs = (m: Msg[]) =>
      update((c) => c.map((x) => (x.id === id ? { ...x, messages: m, updatedAt: Date.now() } : x)));
    setMsgs([...msgs, { role: "assistant", content: "" }]);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    let out = "";
    let toolResults: ToolResult[] = [];
    try {
      const res = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Demo-User-ID": getDemoUserId(),
        },
        body: JSON.stringify({ model, messages: msgs, ...parameters }),
        signal: ac.signal,
      });
      if (!res.ok || !res.body) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `Request failed (${res.status})`);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let chunk;
          try {
            chunk = JSON.parse(data);
          } catch {
            continue;
          }
          if (typeof chunk.error === "string") throw new Error(chunk.error);
          if (Array.isArray(chunk.tool_results)) {
            toolResults = [...toolResults, ...chunk.tool_results];
            const newApprovals = chunk.tool_results
              .map((toolResult: ToolResult) => toPendingApproval(toolResult, id!))
              .filter((approval: PendingApproval | null): approval is PendingApproval =>
                Boolean(approval),
              );
            if (newApprovals.length > 0) {
              setPendingApprovals((current) => [
                ...current,
                ...newApprovals.filter(
                  (approval) =>
                    !current.some((existing) => existing.approvalId === approval.approvalId),
                ),
              ]);
            }
          }
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta || toolResults.length > 0) {
            if (delta) out += delta;
            setMsgs([
              ...msgs,
              {
                role: "assistant",
                content: out,
                ...(toolResults.length > 0 ? { toolResults: [...toolResults] } : {}),
              },
            ]);
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
        if (!out) setMsgs(msgs);
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const decideApproval = async (approved: boolean) => {
    if (!activeApproval || approvalBusy) return;
    setApprovalBusy(true);
    setApprovalError(null);
    try {
      const response = await fetch(
        `${API_BASE}/v1/approvals/${encodeURIComponent(activeApproval.approvalId)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Demo-User-ID": getDemoUserId(),
          },
          body: JSON.stringify({ approved }),
        },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Approval decision failed (${response.status})`);
      }

      update((allChats) =>
        allChats.map((chat) => {
          if (chat.id !== activeApproval.chatId) return chat;
          return {
            ...chat,
            messages: chat.messages.map((message) => ({
              ...message,
              toolResults: message.toolResults?.map((toolResult) => {
                if (toolResult.result.approval_id !== activeApproval.approvalId) {
                  return toolResult;
                }
                return {
                  ...toolResult,
                  result: {
                    ...toolResult.result,
                    approval_status: data.approval_status,
                    ...(data.result ? { refund_result: data.result } : {}),
                  },
                };
              }),
            })),
          };
        }),
      );
      setPendingApprovals((current) =>
        current.filter((approval) => approval.approvalId !== activeApproval.approvalId),
      );
    } catch (e) {
      setApprovalError((e as Error).message);
    } finally {
      setApprovalBusy(false);
    }
  };

  return (
    <div className="flex h-screen bg-background text-foreground">
      <aside className="hidden w-64 flex-col border-r border-border bg-sidebar md:flex">
        <button
          onClick={() => setActiveId(null)}
          className="m-3 flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm hover:bg-sidebar-accent"
        >
          <Plus className="h-4 w-4" /> New chat
        </button>
        <div className="flex-1 overflow-y-auto px-2">
          {chats.map((c) => (
            <div
              key={c.id}
              onClick={() => setActiveId(c.id)}
              className={`group flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-sm ${c.id === activeId ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60"}`}
            >
              <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
              <span className="flex-1 truncate">{c.title}</span>
              <button
                aria-label="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  update((all) => all.filter((x) => x.id !== c.id));
                  if (c.id === activeId) setActiveId(null);
                }}
                className="opacity-0 group-hover:opacity-100"
              >
                <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <button onClick={() => setActiveId(null)} className="md:hidden" aria-label="New chat">
              <Plus className="h-5 w-5" />
            </button>
            <span className="font-semibold">Parley</span>
          </div>
          <div className="flex items-center gap-2">
            <select
              aria-label="Model"
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
                localStorage.setItem("model", e.target.value);
              }}
              className="rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            >
              {models.length === 0 && <option value="">No models found</option>}
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6">
            {!active || active.messages.length === 0 ? (
              <div className="mt-32 text-center">
                <h1 className="text-3xl font-semibold">How can I help today?</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Chats are saved in this browser only.
                </p>
              </div>
            ) : (
              active.messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i} className="group mb-8 flex flex-col items-end">
                    <div className="max-w-[70%] whitespace-pre-wrap break-words rounded-3xl bg-user-bubble px-5 py-2.5 text-[16px] leading-7 text-user-bubble-foreground">
                      {m.content}
                    </div>
                    <div className="mt-1 opacity-0 transition-opacity group-hover:opacity-100">
                      <CopyBtn text={m.content} />
                    </div>
                  </div>
                ) : (
                  <div key={i} className="group mb-8">
                    <div className="prose-chat text-[16px] leading-7">
                      {m.content ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      ) : (
                        <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-foreground" />
                      )}
                    </div>
                    {m.toolResults && m.toolResults.length > 0 && (
                      <div className="mt-4 space-y-3">
                        {m.toolResults.map((toolResult, toolIndex) => (
                          <section
                            key={`${toolResult.name}-${toolIndex}`}
                            className="rounded-lg border border-border bg-muted/40 p-3 text-sm"
                          >
                            <h3 className="font-medium">Tool result: {toolResult.name}</h3>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Arguments: {JSON.stringify(toolResult.arguments)}
                            </p>
                            <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs">
                              {JSON.stringify(toolResult.result, null, 2)}
                            </pre>
                          </section>
                        ))}
                      </div>
                    )}
                    {m.content && (
                      <div className="mt-1 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                        <CopyBtn text={m.content} />
                      </div>
                    )}
                  </div>
                ),
              )
            )}
            {error && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error}
              </div>
            )}
            <div ref={endRef} />
          </div>
        </div>

        <div className="mx-auto w-full max-w-3xl px-4 pb-6">
          <details className="mb-3 rounded-xl border border-border bg-card px-4 py-3">
            <summary className="cursor-pointer text-sm font-medium">Generation settings</summary>
            <div className="mt-4 grid gap-5 sm:grid-cols-2">
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label htmlFor="temperature" className="text-sm">
                    Temperature
                  </label>
                  <input
                    id="temperature"
                    aria-label="Temperature value"
                    type="number"
                    min="0"
                    max="2"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(e.target.value)}
                    className="h-8 w-20 rounded-md border border-input bg-background px-2 text-right text-sm"
                  />
                </div>
                <input
                  aria-label="Temperature slider"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                  className="w-full accent-primary"
                />
                <div className="mt-1 flex justify-between text-xs text-muted-foreground">
                  <span>Focused</span>
                  <span>Creative</span>
                </div>
              </div>
              <div>
                <label htmlFor="max-tokens" className="mb-2 block text-sm">
                  Max tokens
                </label>
                <input
                  id="max-tokens"
                  type="number"
                  min="1"
                  max="32768"
                  step="1"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                  placeholder="Use model default"
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                />
              </div>
            </div>
            <div className="mt-5 border-t border-border pt-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-medium">Additional OpenAI parameters</h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Add any supported chat-completion field; enter its value as JSON (for example,
                    top_p: 0.8).
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    setExtraParameters((current) => [
                      ...current,
                      { id: crypto.randomUUID(), name: "", value: "" },
                    ])
                  }
                  className="inline-flex shrink-0 items-center gap-1 rounded-md border border-input px-2.5 py-1.5 text-xs hover:bg-muted"
                >
                  <Plus className="h-3.5 w-3.5" /> Add field
                </button>
              </div>
              <div className="space-y-2">
                {extraParameters.map((parameter) => (
                  <div key={parameter.id} className="flex items-center gap-2">
                    <input
                      aria-label="Parameter name"
                      value={parameter.name}
                      onChange={(e) =>
                        setExtraParameters((current) =>
                          current.map((item) =>
                            item.id === parameter.id ? { ...item, name: e.target.value } : item,
                          ),
                        )
                      }
                      placeholder="field_name"
                      className="h-9 min-w-0 flex-1 rounded-md border border-input bg-background px-3 text-sm"
                    />
                    <input
                      aria-label="Parameter JSON value"
                      value={parameter.value}
                      onChange={(e) =>
                        setExtraParameters((current) =>
                          current.map((item) =>
                            item.id === parameter.id ? { ...item, value: e.target.value } : item,
                          ),
                        )
                      }
                      placeholder={'JSON value e.g. 0.8, "json_object", ["."]'}
                      className="h-9 min-w-0 flex-[2] rounded-md border border-input bg-background px-3 text-sm"
                    />
                    <button
                      type="button"
                      aria-label={`Remove ${parameter.name || "parameter"}`}
                      onClick={() =>
                        setExtraParameters((current) =>
                          current.filter((item) => item.id !== parameter.id),
                        )
                      }
                      className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </details>
          <div className="flex items-end gap-2 rounded-2xl border border-input bg-card p-2 shadow-sm">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              placeholder="Message Parley…"
              className="max-h-48 flex-1 resize-none bg-transparent px-2 py-2 outline-none"
            />
            {busy ? (
              <button
                onClick={() => abortRef.current?.abort()}
                aria-label="Stop"
                className="rounded-full bg-primary p-2 text-primary-foreground"
              >
                <Square className="h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={send}
                disabled={!input.trim()}
                aria-label="Send"
                className="rounded-full bg-primary p-2 text-primary-foreground disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </main>
      <AlertDialog open={activeApproval !== null}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Approval required: issue mock refund</AlertDialogTitle>
            <AlertDialogDescription>
              The assistant is requesting a refund for order {activeApproval?.orderId}. Review the
              details before deciding.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {activeApproval && (
            <dl className="space-y-2 rounded-md bg-muted p-3 text-sm">
              <div>
                <dt className="font-medium">Amount</dt>
                <dd>${(activeApproval.amountCents / 100).toFixed(2)} USD (mock)</dd>
              </div>
              <div>
                <dt className="font-medium">Reason</dt>
                <dd>{activeApproval.reason}</dd>
              </div>
            </dl>
          )}
          {approvalError && (
            <p role="alert" className="text-sm text-destructive">
              {approvalError}
            </p>
          )}
          <AlertDialogFooter>
            <Button
              type="button"
              variant="outline"
              disabled={approvalBusy}
              onClick={() => decideApproval(false)}
            >
              Reject
            </Button>
            <Button type="button" disabled={approvalBusy} onClick={() => decideApproval(true)}>
              {approvalBusy ? "Submitting…" : "Approve refund"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function toPendingApproval(toolResult: ToolResult, chatId: string): PendingApproval | null {
  const { result } = toolResult;
  if (
    toolResult.name !== "request_refund" ||
    result.approval_status !== "pending" ||
    typeof result.approval_id !== "string" ||
    typeof result.order_id !== "string" ||
    typeof result.amount_cents !== "number" ||
    typeof result.reason !== "string"
  ) {
    return null;
  }
  return {
    approvalId: result.approval_id,
    chatId,
    orderId: result.order_id,
    amountCents: result.amount_cents,
    reason: result.reason,
  };
}

function CopyBtn({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      aria-label="Copy"
      onClick={() => {
        navigator.clipboard.writeText(text);
        setDone(true);
        setTimeout(() => setDone(false), 1500);
      }}
      className="rounded-md p-1.5 text-muted-foreground hover:bg-muted"
    >
      {done ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
    </button>
  );
}
