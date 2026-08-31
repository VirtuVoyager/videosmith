"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listShows, triggerRun, type ShowSummary } from "@/lib/api";

export function TriggerRunForm({
  token,
  onTriggered,
}: {
  token: string;
  onTriggered: () => void;
}) {
  const [brief, setBrief] = useState("");
  const [mode, setMode] = useState<"rhyme" | "topical">("rhyme");
  const [shows, setShows] = useState<ShowSummary[]>([]);
  const [showId, setShowId] = useState(""); // "" = one-off video, no frozen cast
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastProjectId, setLastProjectId] = useState<string | null>(null);

  useEffect(() => {
    listShows(token)
      .then(setShows)
      .catch(() => {
        // Shows are optional (need SS_DB_URL) -- a load failure just means
        // the picker stays empty, not a hard error for the whole form.
      });
  }, [token]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!brief.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const { project_id } = await triggerRun(
        token,
        brief.trim(),
        mode,
        showId || undefined,
      );
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
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-2 rounded border border-neutral-200 bg-white p-4"
    >
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-neutral-700">Start a new video</label>
        <Link href="/shows/new" className="text-sm text-blue-600 underline">
          + Create a show
        </Link>
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          placeholder={
            showId
              ? "This episode's topic, e.g. Bob won't share the couch"
              : "Brief, e.g. counting to five with ducks"
          }
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
        <select
          value={showId}
          onChange={(e) => setShowId(e.target.value)}
          className="rounded border border-neutral-300 px-2 py-2"
        >
          <option value="">no show (fresh cast)</option>
          {shows.map((s) => (
            <option key={s.show_id} value={s.show_id}>
              {s.name}
            </option>
          ))}
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
