import { useEffect, useRef, useState } from "react";

export interface SSEEvent<T = unknown> {
  id: string;
  type: string;
  data: T;
}

export interface UseSSEOptions<T> {
  enabled?: boolean;
  onEvent?: (e: SSEEvent<T>) => void;
  onDone?: () => void;
  onError?: (e: Event) => void;
  /** Event types we care about (defaults to all known types). */
  listen?: string[];
}

const DEFAULT_LISTEN = [
  "log",
  "progress",
  "completed",
  "error",
  "done",
  "heartbeat",
];

/**
 * Subscribe to an SSE endpoint. Auto-closes when `enabled` becomes false or URL
 * changes. `Last-Event-ID` is handled by the browser's EventSource on reconnect.
 */
export function useSSE<T = unknown>(
  url: string | null,
  options: UseSSEOptions<T> = {}
) {
  const { enabled = true, onEvent, onDone, onError, listen = DEFAULT_LISTEN } =
    options;
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onEventRef.current = onEvent;
    onDoneRef.current = onDone;
    onErrorRef.current = onError;
  }, [onEvent, onDone, onError]);

  useEffect(() => {
    if (!enabled || !url) {
      setConnected(false);
      return;
    }

    const es = new EventSource(url);
    setConnected(true);

    const handlers: Record<string, (ev: MessageEvent) => void> = {};
    for (const type of listen) {
      const h = (ev: MessageEvent) => {
        let data: T;
        try {
          data = JSON.parse(ev.data) as T;
        } catch {
          data = ev.data as unknown as T;
        }
        onEventRef.current?.({ id: ev.lastEventId, type, data });
        if (type === "done") {
          onDoneRef.current?.();
          es.close();
          setConnected(false);
        }
      };
      handlers[type] = h;
      es.addEventListener(type, h);
    }

    es.onerror = (e) => {
      onErrorRef.current?.(e);
      // EventSource auto-reconnects on error; we only close if terminal.
    };

    return () => {
      for (const [type, h] of Object.entries(handlers)) {
        es.removeEventListener(type, h);
      }
      es.close();
      setConnected(false);
    };
  }, [url, enabled, listen.join(",")]);

  return { connected };
}
