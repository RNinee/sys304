"use client";

import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { formatAssistantReply, predictTweet, extractUserText } from "@/lib/predict";
import { ThreadWelcome } from "@/app/welcome";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";

const DisasterModelAdapter: ChatModelAdapter = {
  async run({ messages, abortSignal }) {
    const last = [...messages].reverse().find((message) => message.role === "user");
    const text = extractUserText(last?.content ?? []);
    if (!text) {
      return {
        content: [{ type: "text", text: "Type a tweet to classify." }],
      };
    }
    try {
      const result = await predictTweet({ text }, { signal: abortSignal });
      return {
        content: [{ type: "text", text: formatAssistantReply(result) }],
      };
    } catch (error) {
      if (abortSignal.aborted) throw error;
      const message = error instanceof Error ? error.message : String(error);
      return {
        content: [{ type: "text", text: `Could not classify that tweet. ${message}` }],
      };
    }
  },
};

export const Assistant = () => {
  const runtime = useLocalRuntime(DisasterModelAdapter);

  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <div className="h-dvh">
          <Thread components={{ Welcome: ThreadWelcome }} />
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
};
