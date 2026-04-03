import type { ReactNode } from "react";

interface ArenaLayoutProps {
  stage: ReactNode;
  controls: ReactNode;
  timeline?: ReactNode;
  sidePanel?: ReactNode;
  utilityRail?: ReactNode;
  timelineExpanded?: boolean;
  sidePanelOpen?: boolean;
  layoutMode?: "default" | "wide-stage";
}

export function ArenaLayout({
  stage,
  controls,
  timeline,
  sidePanel,
  utilityRail,
  timelineExpanded = false,
  sidePanelOpen = false,
  layoutMode = "default",
}: ArenaLayoutProps) {
  const className = [
    "arena-layout",
    layoutMode === "wide-stage" ? "arena-layout--wide-stage" : "",
    timelineExpanded ? "arena-layout--timeline-open" : "arena-layout--timeline-closed",
    sidePanelOpen ? "arena-layout--side-panel-open" : "",
  ]
    .filter((value) => value !== "")
    .join(" ");

  return (
    <section className={className} aria-label="Arena workspace theater">
      <div className="arena-layout__controls">{controls}</div>
      <div className="arena-layout__theater">
        <div className="arena-layout__stage">{stage}</div>
        {utilityRail ? (
          <nav className="arena-layout__utility-rail" aria-label="Session utility rail">
            {utilityRail}
          </nav>
        ) : null}
        {sidePanel ? <aside className="arena-layout__side-panel">{sidePanel}</aside> : null}
      </div>
      {timeline ? <div className="arena-layout__timeline">{timeline}</div> : null}
    </section>
  );
}
