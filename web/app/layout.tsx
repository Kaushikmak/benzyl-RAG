import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Obsidian RAG",
  description: "Simple Next.js UI for Obsidian RAG",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
