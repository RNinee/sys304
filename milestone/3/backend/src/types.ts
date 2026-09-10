export type PredictRequest = {
  text: string;
  keyword?: string;
};

export type PredictResponse = {
  target: 0 | 1;
  label: "disaster" | "not disaster";
  confidence: number;
  probabilities: {
    not_disaster: number;
    disaster: number;
  };
  model: string;
  input: string;
  cached?: boolean;
  batch_size?: number;
};

export type HealthResponse = {
  status: "ok";
  model: string;
  role: "api" | "infer-worker";
  infer?: "up" | "down";
  cache?: "up" | "down" | "off";
  optimizations?: string[];
};
