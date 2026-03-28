import type { ReactNode } from "react";

interface ArenaLayoutProps {
  stage: ReactNode;
  controls: ReactNode;
  timeline: ReactNode;
  sidePanel: ReactNode;
}

export function ArenaLayout({
  stage,
  controls,
  timeline,
  sidePanel,
}: ArenaLayoutProps) {
  return (
    <section className="arena-layout" aria-label="Arena workspace">
      <div className="arena-layout__controls">{controls}</div>
      <div className="arena-layout__stage">{stage}</div>
      <aside className="arena-layout__side-panel">{sidePanel}</aside>
      <div className="arena-layout__timeline">{timeline}</div>
    </section>
  );
}
