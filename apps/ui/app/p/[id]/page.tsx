"use client";

import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import {
  approveProject,
  downloadAsset,
  getProject,
  rejectProject,
  type ProjectDetail,
} from "@/lib/api";
import { TokenGate } from "../../TokenGate";

function VerdictBadge({ verdict }: { verdict: string }) {
  const color =
    verdict === "pass"
      ? "bg-green-100 text-green-800"
      : verdict === "retry"
        ? "bg-amber-100 text-amber-800"
        : "bg-red-100 text-red-800";
  return <span className={`rounded-full px-2 py-1 text-xs font-medium ${color}`}>{verdict}</span>;
}

function ProjectDetailView({ token, projectId }: { token: string; projectId: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const reload = useCallback(() => {
    getProject(token, projectId)
      .then(setProject)
      .catch((e) => setError(String(e)));
  }, [token, projectId]);

  useEffect(reload, [reload]);

  const finalVideo = project?.assets.find((a) => a.kind === "final_video");
  const thumbnail = project?.assets.find((a) => a.kind === "thumbnail");

  async function handleApprove() {
    setBusy(true);
    try {
      setProject(await approveProject(token, projectId));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleReject() {
    setBusy(true);
    try {
      setProject(await rejectProject(token, projectId));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleDownload() {
    if (!finalVideo) return;
    setDownloading(true);
    try {
      const filename = `${(project?.title ?? project?.brief ?? "storysmith-video").replace(/[^a-z0-9]+/gi, "_")}.mp4`;
      await downloadAsset(token, finalVideo.presigned_url, filename);
    } catch (e) {
      setError(String(e));
    } finally {
      setDownloading(false);
    }
  }

  if (error) return <p className="text-red-600">{error}</p>;
  if (!project) return <p>Loading…</p>;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">{project.title ?? project.brief}</h1>
        <p className="text-neutral-500">
          {project.mode} · status: {project.status} · ${project.total_cost_usd.toFixed(2)}
        </p>
      </div>

      {finalVideo ? (
        <div className="flex flex-col items-start gap-2">
          {/* Local storage's presigned_url isn't a browser-fetchable URL
              (see storage_local.py) -- this preview may not actually play.
              The download button below goes through the authenticated
              /assets/view proxy instead, which works regardless of
              storage backend. */}
          <video
            controls
            poster={thumbnail?.presigned_url}
            src={finalVideo.presigned_url}
            className="w-full max-w-sm rounded border border-neutral-200"
          />
          <button
            onClick={handleDownload}
            disabled={downloading}
            className="rounded border border-neutral-300 px-4 py-2 text-sm disabled:opacity-50"
          >
            {downloading ? "Downloading…" : "Download video"}
          </button>
        </div>
      ) : (
        <p className="text-neutral-500">No final cut yet.</p>
      )}

      {project.status === "review" && (
        <div className="flex gap-3">
          <button
            onClick={handleApprove}
            disabled={busy}
            className="rounded bg-green-700 px-4 py-2 text-white disabled:opacity-50"
          >
            Approve &amp; publish
          </button>
          <button
            onClick={handleReject}
            disabled={busy}
            className="rounded bg-red-700 px-4 py-2 text-white disabled:opacity-50"
          >
            Reject
          </button>
        </div>
      )}

      {project.published_url && (
        <a href={project.published_url} className="text-blue-600 underline">
          {project.published_url}
        </a>
      )}

      <div>
        <h2 className="mb-2 text-lg font-medium">QA reports</h2>
        <ul className="flex flex-col gap-2">
          {project.qa_reports.map((r, i) => (
            <li key={i} className="rounded border border-neutral-200 bg-white p-3">
              <div className="flex items-center justify-between">
                <span className="font-medium">
                  {r.scene_index === null ? "audio" : `scene ${r.scene_index}`}
                </span>
                <VerdictBadge verdict={r.verdict} />
              </div>
              {r.critique && <p className="mt-1 text-sm text-neutral-600">{r.critique}</p>}
              {r.safety_flags.length > 0 && (
                <p className="mt-1 text-sm text-red-600">flags: {r.safety_flags.join(", ")}</p>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  return (
    <main>
      <TokenGate>{(token) => <ProjectDetailView token={token} projectId={params.id} />}</TokenGate>
    </main>
  );
}
