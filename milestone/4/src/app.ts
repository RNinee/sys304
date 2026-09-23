import { type AppOptions, createApp } from "../../3/backend/src/app";
import { loadTrainBaseline, type TrainBaseline } from "./baseline";
import { RequestLog } from "./log";
import { renderPrometheus } from "./prometheus";

export type MonitorOptions = AppOptions & {
  log?: RequestLog;
  baseline?: TrainBaseline;
  onReload?: () => Promise<void>;
};

export function createMonitorApp(options: MonitorOptions = {}) {
  const log = options.log ?? new RequestLog();
  const baseline = options.baseline ?? loadTrainBaseline();
  const token = process.env.ADMIN_TOKEN;

  const core = createApp({
    infer: options.infer,
    cache: options.cache,
    onLogged: (event) => {
      log.insert(event);
      options.onLogged?.(event);
    },
  });

  return core
    .get("/prometheus", () => {
      return new Response(renderPrometheus(log.metricsRows(), baseline), {
        headers: { "Content-Type": "text/plain; version=0.0.4; charset=utf-8" },
      });
    })
    .post("/admin/reload", async ({ request, set }) => {
      if (token && request.headers.get("x-admin-token") !== token) {
        set.status = 401;
        return { error: "unauthorized" };
      }
      if (!options.onReload) {
        set.status = 409;
        return { error: "this process does not manage the infer worker" };
      }
      await options.onReload();
      return { restarted: true };
    });
}
