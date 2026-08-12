import type { components } from "./api-types";

export type ProjectSummary = components["schemas"]["ProjectSummary"];
export type ProjectDetail = components["schemas"]["ProjectDetail"];

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
