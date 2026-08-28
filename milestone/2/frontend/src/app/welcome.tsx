"use client";

import { Button } from "@/components/ui/button";
import { useAui } from "@assistant-ui/react";

const EXAMPLES = [
  {
    label: "Forest fire",
    text: "Forest fire near La Ronge Sask. Canada",
  },
  {
    label: "Earthquake",
    text: "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all",
  },
  {
    label: "Movie metaphor",
    text: "The plot twist was a total disaster, I loved every second of it",
  },
] as const;

export function ThreadWelcome() {
  const aui = useAui();

  return (
    <div className="aui-thread-welcome-root mb-6 flex flex-col items-center px-4 text-center">
      <p className="text-muted-foreground mb-2 text-xs tracking-[0.2em] uppercase">
        SYS-304 · Milestone 2
      </p>
      <h1
        data-testid="welcome-title"
        className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 animate-in fill-mode-both text-2xl font-medium tracking-tight duration-200"
      >
        Disaster tweet classifier
      </h1>
      <p className="text-muted-foreground mt-3 max-w-md text-sm leading-6">
        Paste a tweet, not a greeting. The model was trained on Kaggle
        disaster tweets (real event vs joke/metaphor), so short strings like
        “hello” get overconfident scores.
      </p>
      <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
        {EXAMPLES.map((example) => (
          <Button
            key={example.label}
            type="button"
            variant="ghost"
            className="border-border/60 h-auto rounded-full border px-3.5 py-1.5 text-sm font-normal"
            onClick={() => {
              aui.thread.append({
                content: [{ type: "text", text: example.text }],
              });
            }}
          >
            {example.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
