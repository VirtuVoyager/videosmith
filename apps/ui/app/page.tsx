"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { listProjects, type ProjectSummary } from "@/lib/api";
import { TokenGate } from "./TokenGate";
import { TriggerRunForm } from "./TriggerRunForm";

const STATUS_COLOR: Record<string, string> = {
  review: "bg-amber-100 text-amber-800",
  published: "bg-green-100 text-green-800",
  rejected: "bg-neutral-200 text-neutral-700",
  failed: "bg-red-100 text-red-800",
  budget_abort: "bg-red-100 text-red-800",
};

function ProjectList({ token }: { token: string }) {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    listProjects(token)
      .then(setProjects)
      .catch((e) => setError(String(e)));
  }, [token]);

  useEffect(reload, [reload]);

  return (
    <div className="flex flex-col gap-6">
      <TriggerRunForm token={token} onTriggered={reload} />

      {error && <p className="text-red-600">{error}</p>}
      {!error && !projects && <p>Loading…</p>}
      {!error && projects && projects.length === 0 && (
        <p className="text-neutral-500">No projects yet.</p>
      )}
      {!error && projects && projects.length > 0 && (
        <ul className="divide-y divide-neutral-200 rounded border border-neutral-200 bg-white">
          {projects.map((p) => (
            <li key={p.project_id}>
              <Link
                href={`/p/${p.project_id}`}
                className="flex items-center justify-between px-4 py-3 hover:bg-neutral-50"
              >
                <div>
                  <div className="font-medium">{p.title ?? p.brief}</div>
                  <div className="text-sm text-neutral-500">
                    {p.mode} · ${p.total_cost_usd.toFixed(2)} ·{" "}
                    {new Date(p.updated_at).toLocaleString()}
                  </div>
                </div>
                <span
                  className={`rounded-full px-2 py-1 text-xs font-medium ${STATUS_COLOR[p.status] ?? "bg-neutral-100 text-neutral-700"}`}
                >
                  {p.status}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">StorySmith Review Console</h1>
      <TokenGate>{(token) => <ProjectList token={token} />}</TokenGate>
    </main>
  );
}
