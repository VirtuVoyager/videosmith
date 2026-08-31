import type { components } from "./api-types";

export type ProjectSummary = components["schemas"]["ProjectSummary"];
export type ProjectDetail = components["schemas"]["ProjectDetail"];
export type ShowSummary = components["schemas"]["ShowSummary"];
export type ShowDetail = components["schemas"]["ShowDetail"];
export type CharacterInput = components["schemas"]["CharacterInput"];
export type CreateShowRequest = components["schemas"]["CreateShowRequest"];

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

export async function listProjects(token: string): Promise<ProjectSummary[]> {
  const res = await fetch(`${API_BASE}/projects`, {
    headers: authHeaders(token),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`GET /projects failed: ${res.status}`);
  return res.json();
}

export async function getProject(token: string, projectId: string): Promise<ProjectDetail> {
  const res = await fetch(`${API_BASE}/projects/${projectId}`, {
    headers: authHeaders(token),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`GET /projects/${projectId} failed: ${res.status}`);
  return res.json();
}

export async function approveProject(token: string, projectId: string): Promise<ProjectDetail> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/approve`, {
    method: "POST",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`approve failed: ${res.status}`);
  return res.json();
}

export async function rejectProject(token: string, projectId: string): Promise<ProjectDetail> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/reject`, {
    method: "POST",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`reject failed: ${res.status}`);
  return res.json();
}

export async function triggerRun(
  token: string,
  brief: string,
  mode: "rhyme" | "topical",
  showId?: string,
): Promise<{ project_id: string }> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ brief, mode, show_id: showId || null }),
  });
  if (!res.ok) throw new Error(`trigger run failed: ${res.status}`);
  return res.json();
}

export async function listShows(token: string): Promise<ShowSummary[]> {
  const res = await fetch(`${API_BASE}/shows`, { headers: authHeaders(token), cache: "no-store" });
  if (!res.ok) throw new Error(`GET /shows failed: ${res.status}`);
  return res.json();
}

export async function createShow(
  token: string,
  body: CreateShowRequest,
): Promise<ShowDetail> {
  const res = await fetch(`${API_BASE}/shows`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(`create show failed: ${res.status} ${detail?.detail ?? ""}`);
  }
  return res.json();
}

/** Fetches an asset through the authenticated proxy (GET /assets/view) and
 * returns a browser object URL for it -- neither <img src> nor a plain
 * <a href> can carry the bearer header, so this is the only way to load a
 * storage asset (avatar, final video) into the page at all. Callers should
 * revoke the returned URL (URL.revokeObjectURL) when done with it, e.g. on
 * unmount, to avoid leaking memory. */
export async function fetchAssetObjectUrl(token: string, assetUri: string): Promise<string> {
  const res = await fetch(`${API_BASE}/assets/view?uri=${encodeURIComponent(assetUri)}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`GET /assets/view failed: ${res.status}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

/** Fetches the asset then immediately triggers a browser "save as" for it --
 * used for the final-video download button. */
export async function downloadAsset(
  token: string,
  assetUri: string,
  filename: string,
): Promise<void> {
  const objectUrl = await fetchAssetObjectUrl(token, assetUri);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}
