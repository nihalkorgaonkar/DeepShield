import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepShield — Speech Deepfake Detection Research Platform",
  description:
    "Empirical Comparison of Vision Transformer (ViT-B/16) and CNN Baseline (EfficientNet-B0) on ASVspoof Benchmarks under Strict Academic Integrity Protocols.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#090d16] text-[#f1f5f9] antialiased selection:bg-sky-500/30 selection:text-sky-200 min-h-screen">
        {children}
      </body>
    </html>
  );
}
