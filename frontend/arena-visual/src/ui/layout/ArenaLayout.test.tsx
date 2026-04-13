import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render, screen, within } from "@testing-library/react";

import { ArenaLayout } from "./ArenaLayout";

describe("ArenaLayout", () => {
  it("keeps the utility rail and side panel outside the theater so host chrome does not cover the stage", () => {
    const { container } = render(
      <ArenaLayout
        layoutMode="wide-stage"
        sidePanelOpen={true}
        controls={<div>Controls</div>}
        stage={<div>Stage</div>}
        utilityRail={<button type="button">Chat</button>}
        sidePanel={<div>Panel</div>}
      />,
    );

    const workspace = container.querySelector<HTMLElement>(".arena-layout");
    const theater = container.querySelector<HTMLElement>(".arena-layout__theater");
    const utilityRail = container.querySelector<HTMLElement>(".arena-layout__utility-rail");
    const sidePanel = container.querySelector<HTMLElement>(".arena-layout__side-panel");

    expect(workspace).not.toBeNull();
    expect(theater).not.toBeNull();
    expect(utilityRail).not.toBeNull();
    expect(sidePanel).not.toBeNull();

    expect(workspace).toContainElement(theater);
    expect(workspace).toContainElement(utilityRail);
    expect(workspace).toContainElement(sidePanel);
    expect(theater).not.toContainElement(utilityRail);
    expect(theater).not.toContainElement(sidePanel);
    expect(within(utilityRail as HTMLElement).getByRole("button", { name: "Chat" })).toBeInTheDocument();
    expect(sidePanel).toHaveTextContent("Panel");
    expect(screen.getByText("Stage")).toBeInTheDocument();
  });

  it("keeps the session side panel in the workspace grid instead of the legacy outer side rail", () => {
    const baseCss = readFileSync(resolve(__dirname, "../../styles/base.css"), "utf8");

    expect(baseCss).toMatch(
      /\.arena-layout__utility-rail\s*\{[^}]*align-self:\s*start;/s,
    );
    expect(baseCss).toMatch(
      /\.arena-layout__side-panel\s*\{[^}]*grid-area:\s*auto;[^}]*align-self:\s*stretch;/s,
    );
    expect(baseCss).toMatch(
      /\.arena-layout--side-panel-open\s+\.arena-layout__workspace\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s+auto\s+minmax\(18rem,\s*25rem\);/s,
    );
  });
});
