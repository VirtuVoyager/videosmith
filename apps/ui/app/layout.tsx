import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StorySmith Review Console",
  description: "Approve or reject nightly StorySmith video projects.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-neutral-50 text-neutral-900 antialiased">
        <div className="mx-auto max-w-4xl px-4 py-8">{children}</div>
      </body>
    </html>
  );
}
