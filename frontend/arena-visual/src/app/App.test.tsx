import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

vi.mock("./routes/SessionPage", () => ({
  SessionPage: () => <h1>Session workspace mock</h1>,
}));

import { App } from "./App";

describe("App", () => {
  it("renders the empty host route", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("heading", { name: /arena visual host/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/open a session artifact to start exploring/i),
    ).toBeInTheDocument();
  });

  it("keeps the session route mounted inside the app shell", () => {
    render(
      <MemoryRouter initialEntries={["/sessions/demo-session"]}>
        <App />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("heading", { name: /session workspace mock/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /home/i })).toBeInTheDocument();
  });

  it("redirects unknown routes back to the host home", () => {
    render(
      <MemoryRouter initialEntries={["/missing-route"]}>
        <App />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("heading", { name: /arena visual host/i }),
    ).toBeInTheDocument();
  });
});
