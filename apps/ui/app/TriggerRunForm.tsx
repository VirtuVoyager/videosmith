"use client";

import { useState } from "react";
import { triggerRun } from "@/lib/api";

export function TriggerRunForm({
  token,
  onTriggered,
}: {
  token: string;
  onTriggered: () => void;
}) {
  const [brief, setBrief] = useState("");
  const [mode, setMode] = useState<"rhyme" | "topical">("rhyme");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastProjectId, setLastProjectId] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!brief.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const { project_id } = await triggerRun(token, brief.trim(), mode);
      setLastProjectId(project_id);
      setBrief("");
      onTriggered();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 rounded border border-neutral-200 bg-white p-4">
      <label className="text-sm font-medium text-neutral-700">Start a new video</label>
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Brief, e.g. counting to five with ducks"
          value={brief}
          onChange={(e) => setBrief(e.target.value)}
          className="flex-1 rounded border border-neutral-300 px-3 py-2"
        />
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as "rhyme" | "topical")}
          className="rounded border border-neutral-300 px-2 py-2"
        >
          <option value="rhyme">rhyme</option>
          <option value="topical">topical</option>
        </select>
        <button
          type="submit"
          disabled={busy || !brief.trim()}
          className="rounded bg-neutral-900 px-4 py-2 text-white disabled:opacity-50"
        >
          {busy ? "Starting…" : "Run"}
        </button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {lastProjectId && !error && (
        <p className="text-sm text-neutral-500">
          Started project {lastProjectId} — it&apos;ll appear below once the first node completes.
        </p>
      )}
    </form>
  );
}
