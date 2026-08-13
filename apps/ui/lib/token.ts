"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "storysmith_api_token";

/** Single-operator private console (§7) -- the bearer token lives in the
 * browser's localStorage, entered once, rather than a full login/session
 * system. Not meant for a multi-tenant/public deployment. */
export function useApiToken(): [string, (token: string) => void] {
  const [token, setToken] = useState("");

  useEffect(() => {
    setToken(localStorage.getItem(STORAGE_KEY) ?? "");
  }, []);

  const update = (next: string) => {
    localStorage.setItem(STORAGE_KEY, next);
    setToken(next);
  };

  return [token, update];
}
