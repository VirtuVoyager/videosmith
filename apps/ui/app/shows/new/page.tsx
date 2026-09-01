"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { createShow, fetchAssetObjectUrl, type CharacterInput, type ShowDetail } from "@/lib/api";
import { TokenGate } from "../../TokenGate";

// Kokoro-82M's named voice pack (settings.tts_voice's default is af_bella) --
// picking one per character here just sets CharacterRef.voice_id; there's no
// server-side validation of the id, so keeping this list in sync with the
// actual model's voices is a documentation concern, not a correctness one.
const VOICE_OPTIONS = [
  "af_bella",
  "af_nicole",
  "af_sky",
  "am_adam",
  "am_michael",
  "bf_emma",
  "bf_isabella",
  "bm_george",
  "bm_lewis",
];

function emptyCharacter(): CharacterInput {
  return { name: "", description: "", personality: "", voice_id: VOICE_OPTIONS[0] };
}

function AvatarImage({ token, assetUri, alt }: { token: string; assetUri: string; alt: string }) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let url: string | null = null;
    fetchAssetObjectUrl(token, assetUri).then((u) => {
      if (cancelled) {
        URL.revokeObjectURL(u);
        return;
      }
      url = u;
      setObjectUrl(u);
    });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [token, assetUri]);

  if (!objectUrl) return <div className="h-32 w-32 rounded bg-neutral-100" />;
  return (
    // eslint-disable-next-line @next/next/no-img-element -- object URL, not a static/remote asset Next can optimize
    <img src={objectUrl} alt={alt} className="h-32 w-32 rounded object-cover" />
  );
}

function CreateShowForm({ token }: { token: string }) {
  const [showId, setShowId] = useState("");
  const [name, setName] = useState("");
  const [artStyle, setArtStyle] = useState(
    "polished realistic 3D Disney-Pixar style CG animation, cinematic lighting, expressive stylized proportions, subsurface skin scattering",
  );
  const [characters, setCharacters] = useState<CharacterInput[]>([emptyCharacter()]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<ShowDetail | null>(null);

  function updateCharacter(index: number, patch: Partial<CharacterInput>) {
    setCharacters((prev) => prev.map((c, i) => (i === index ? { ...c, ...patch } : c)));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const show = await createShow(token, {
        show_id: showId.trim(),
        name: name.trim(),
        art_style: artStyle.trim(),
        // Match the API's own Pydantic defaults explicitly -- the generated
        // client type keeps default-valued fields required even though the
        // server doesn't (openapi-typescript documents fields with a
        // `default` as non-optional in the type, since JSON Schema's
        // `required` list is silent on them either way).
        palette: [],
        mood: "cheerful",
        tempo_bpm: 100,
        aspect_ratio: "9:16",
        resolution: "1080x1920",
        pacing_rules: "",
        characters,
      });
      setCreated(show);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  if (created) {
    return (
      <div className="flex flex-col gap-4">
        <p className="text-green-700">
          Show <strong>{created.name}</strong> created — this cast is now frozen and reusable
          for every future episode via show_id <code>{created.show_id}</code>.
        </p>
        <div className="flex gap-4">
          {created.characters.map((c) => (
            <div key={c.name} className="flex flex-col items-center gap-2">
              <AvatarImage token={token} assetUri={c.image_asset_uri} alt={c.name} />
              <span className="text-sm font-medium">{c.name}</span>
            </div>
          ))}
        </div>
        <Link href="/" className="text-blue-600 underline">
          Back to projects — start an episode against this show
        </Link>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <div className="flex flex-col gap-3 rounded border border-neutral-200 bg-white p-4">
        <label className="flex flex-col gap-1 text-sm">
          Show ID (slug used to reuse this cast later, e.g. <code>bob-and-miko</code>)
          <input
            required
            value={showId}
            onChange={(e) => setShowId(e.target.value)}
            className="rounded border border-neutral-300 px-3 py-2"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Show name
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="rounded border border-neutral-300 px-3 py-2"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Art style
          <input
            required
            value={artStyle}
            onChange={(e) => setArtStyle(e.target.value)}
            className="rounded border border-neutral-300 px-3 py-2"
          />
        </label>
      </div>

      <div className="flex flex-col gap-4">
        <h2 className="text-lg font-medium">Characters</h2>
        {characters.map((character, i) => (
          <div key={i} className="flex flex-col gap-2 rounded border border-neutral-200 bg-white p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-neutral-500">Character {i + 1}</span>
              {characters.length > 1 && (
                <button
                  type="button"
                  onClick={() => setCharacters((prev) => prev.filter((_, idx) => idx !== i))}
                  className="text-sm text-red-600"
                >
                  Remove
                </button>
              )}
            </div>
            <label className="flex flex-col gap-1 text-sm">
              Name
              <input
                required
                value={character.name}
                onChange={(e) => updateCharacter(i, { name: e.target.value })}
                className="rounded border border-neutral-300 px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              Physical description (feeds the avatar image directly)
              <textarea
                required
                value={character.description}
                onChange={(e) => updateCharacter(i, { description: e.target.value })}
                className="rounded border border-neutral-300 px-3 py-2"
                rows={2}
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              Personality &amp; quirks (guides how they talk, not how they look)
              <textarea
                value={character.personality}
                onChange={(e) => updateCharacter(i, { personality: e.target.value })}
                className="rounded border border-neutral-300 px-3 py-2"
                rows={2}
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              Voice
              <select
                value={character.voice_id ?? VOICE_OPTIONS[0]}
                onChange={(e) => updateCharacter(i, { voice_id: e.target.value })}
                className="rounded border border-neutral-300 px-3 py-2"
              >
                {VOICE_OPTIONS.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </div>
        ))}
        <button
          type="button"
          onClick={() => setCharacters((prev) => [...prev, emptyCharacter()])}
          className="self-start rounded border border-neutral-300 px-3 py-1 text-sm"
        >
          + Add character
        </button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      <button
        type="submit"
        disabled={busy}
        className="self-start rounded bg-neutral-900 px-4 py-2 text-white disabled:opacity-50"
      >
        {busy ? "Generating avatars…" : "Create show"}
      </button>
    </form>
  );
}

export default function NewShowPage() {
  return (
    <main className="flex flex-col gap-6">
      <div>
        <Link href="/" className="text-sm text-blue-600 underline">
          ← Projects
        </Link>
        <h1 className="text-2xl font-semibold">Create a show</h1>
        <p className="text-neutral-500">
          Describe a fixed cast once — their avatars get generated and locked in, and every
          future episode reuses this exact cast against a new topic.
        </p>
      </div>
      <TokenGate>{(token) => <CreateShowForm token={token} />}</TokenGate>
    </main>
  );
}
