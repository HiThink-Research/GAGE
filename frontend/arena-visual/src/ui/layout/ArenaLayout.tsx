import type { ReactNode } from "react";

interface ArenaLayoutProps {
  stage: ReactNode;
  controls: ReactNode;
  timeline: ReactNode;
  sidePanel: ReactNode;
  layoutMode?: "default" | "wide-stage";
}

export function ArenaLayout({
  stage,
  controls,
  timeline,
  sidePanel,
  layoutMode = "default",
}: ArenaLayoutProps) {
  const className = [
    "arena-layout",
    layoutMode === "wide-stage" ? "arena-layout--wide-stage" : "",
  ]
    .filter((value) => value !== "")
    .join(" ");

  return (
    <section className={className} aria-label="Arena workspace">
      <div className="arena-layout__controls">{controls}</div>
      <div className="arena-layout__stage">{stage}</div>
      <aside className="arena-layout__side-panel">{sidePanel}</aside>
      <div className="arena-layout__timeline">{timeline}</div>
    </section>
  );
}
