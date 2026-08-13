"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listProjects, type ProjectSummary } from "@/lib/api";
import { TokenGate } from "./TokenGate";

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

  useEffect(() => {
    listProjects(token)
      .then(setProjects)
      .catch((e) => setError(String(e)));
  }, [token]);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!projects) return <p>Loading…</p>;
  if (projects.length === 0) return <p className="text-neutral-500">No projects yet.</p>;

  return (
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
                {p.mode} · ${p.total_cost_usd.toFixed(2)} · {new Date(p.updated_at).toLocaleString()}
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
