import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

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
});
