import { Navigate, NavLink, Route, Routes } from "react-router-dom";

import { SessionPage } from "./routes/SessionPage";

function HostHome() {
  return (
    <main className="app-shell__body">
      <section className="hero-panel">
        <p className="eyebrow">Independent Arena Workspace</p>
        <h1>Arena Visual Host</h1>
        <p className="hero-copy">
          Open a live or recorded session to inspect timelines, rich game
          stages, observer views, and action diagnostics from one workspace.
        </p>
      </section>

      <section className="panel-grid" aria-label="Host placeholders">
        <article className="panel-card">
          <h2>Recent Sessions</h2>
          <p>
            Session artifacts can be opened directly through their viewer URLs
            while the host shell grows toward a browsable session index.
          </p>
        </article>
        <article className="panel-card">
          <h2>Workspace Regions</h2>
          <p>
            Stage, timeline, controls, and shared inspector panels already run
            through the unified arena gateway and plugin host.
          </p>
        </article>
      </section>
    </main>
  );
}

export function App() {
  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <div>
          <p className="eyebrow">Arena Visual</p>
          <p className="app-shell__title">Unified session workspace</p>
        </div>
        <nav className="app-shell__nav" aria-label="Primary">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              isActive ? "app-shell__nav-link is-active" : "app-shell__nav-link"
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/sessions/demo-session"
            className={({ isActive }) =>
              isActive ? "app-shell__nav-link is-active" : "app-shell__nav-link"
            }
          >
            Demo session
          </NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<HostHome />} />
        <Route path="/sessions/:sessionId" element={<SessionPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
