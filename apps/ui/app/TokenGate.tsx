"use client";

import { useState } from "react";
import { useApiToken } from "@/lib/token";

export function TokenGate({ children }: { children: (token: string) => React.ReactNode }) {
  const [token, setToken] = useApiToken();
  const [draft, setDraft] = useState("");

  if (!token) {
    return (
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setToken(draft);
        }}
      >
        <input
          type="password"
          placeholder="API bearer token"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          className="flex-1 rounded border border-neutral-300 px-3 py-2"
        />
        <button type="submit" className="rounded bg-neutral-900 px-4 py-2 text-white">
          Save
        </button>
      </form>
    );
  }

  return <>{children(token)}</>;
}
