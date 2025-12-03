import type { ChatModelAdapter, ChatModelRunOptions } from "@assistant-ui/react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

/**
 * ChatModelAdapter for Book Recommendations
 * Connects to FastAPI backend with streaming support
 * 
 * Usage in your runtime provider:
 * ```tsx
 * import { useLocalRuntime, AssistantRuntimeProvider } from "@assistant-ui/react";
 * import { MyModelAdapter } from "./api/chat/route";
 * 
 * export function MyRuntimeProvider({ children }) {
 *   const runtime = useLocalRuntime(MyModelAdapter);
 *   return (
 *     <AssistantRuntimeProvider runtime={runtime}>
 *       {children}
 *     </AssistantRuntimeProvider>
 *   );
 * }
 * ```
 */
export const MyModelAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }: ChatModelRunOptions) {
    // Extract text from the last message
    const lastMessage = messages.at(-1);
    const query = lastMessage?.content
      .filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join(" ") ?? "";

    console.log("📚 Book recommendation query:", query);

    // Build conversation history from previous messages
    const history = messages.slice(0, -1).map((msg) => ({
      role: msg.role,
      content: msg.content
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join(" "),
    }));

    const requestBody = {
      message: query,
      history,
      max_tokens: 150,
      temperature: 0.4,
      top_p: 0.8,
    };

    const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
      signal: abortSignal,
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error("No response body");
    }

    // Stream the response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        if (chunk) {
          accumulatedText += chunk;
          
          // Yield accumulated text as content
          yield {
            content: [{ type: "text" as const, text: accumulatedText }],
          };
        }
      }
    } finally {
      reader.releaseLock();
    }
  },
};