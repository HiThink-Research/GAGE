"""Shared utilities for arena visualizers."""

from __future__ import annotations

from typing import Optional


def build_board_interaction_js(
    board_container_id: str,
    move_input_id: str,
    submit_button_id: str,
    *,
    enable_click: bool = True,
    refresh_button_id: Optional[str] = None,
    refresh_interval_ms: Optional[int] = None,
) -> str:
    """Build JS snippet to forward board clicks into a textbox submission.

    Args:
        board_container_id: HTML id for the board container.
        move_input_id: HTML id for the input textbox container.
        submit_button_id: HTML id for the submit button container.
        enable_click: Whether to enable click-to-move behavior.
        refresh_button_id: Optional id of a refresh button to auto-trigger.
        refresh_interval_ms: Optional polling interval in milliseconds.

    Returns:
        JavaScript snippet executed by Gradio.
    """

    helper_block = """
  console.log("[Gomoku] JS Interaction Script Loaded");
  const getRoot = () => {
    const app = document.querySelector("gradio-app");
    if (app && app.shadowRoot) {
      return app.shadowRoot;
    }
    return document;
  };
  const findDeep = (root, selector) => {
    if (!root || !selector) {
      return null;
    }
    if (root.querySelector) {
      const direct = root.querySelector(selector);
      if (direct) {
        return direct;
      }
    }
    if (!root.querySelectorAll) {
      return null;
    }
    const nodes = root.querySelectorAll("*");
    for (const node of nodes) {
      if (node && node.shadowRoot) {
        const found = findDeep(node.shadowRoot, selector);
        if (found) {
          return found;
        }
      }
    }
    return null;
  };
  const find = (selector) => {
    const root = getRoot();
    return findDeep(root, selector) || findDeep(document, selector);
  };
  const resolveControl = (containerSelector, controlSelector) => {
    const container = find(containerSelector);
    if (!container) {
      console.log(`[Gomoku] Container not found: ${containerSelector}`);
      return null;
    }
    if (container.matches && container.matches(controlSelector)) {
      return container;
    }
    return (
      findDeep(container, controlSelector) ||
      (container.shadowRoot ? findDeep(container.shadowRoot, controlSelector) : null)
    );
  };
  const readCoord = (node) => {
    if (!node || !node.getAttribute) {
      return null;
    }
    return (
      node.getAttribute("data-coord") ||
      node.getAttribute("aria-label") ||
      node.getAttribute("title")
    );
  };
  if (!window.__gomoku_hide_errors__) {
    window.__gomoku_hide_errors__ = true;
    const suppressErrors = () => {
      const selectors = [
        ".toast-wrap",
        ".toast",
        ".toast-container",
        ".gradio-error",
        ".error",
        ".error-message",
        ".error-box",
        ".error-panel",
        ".notification",
        ".alert",
        "[role='alert']",
        "[aria-live='assertive']",
        "[aria-live='polite']",
      ];
      const roots = [getRoot(), document];
      for (const root of roots) {
        if (!root || !root.querySelectorAll) {
          continue;
        }
        for (const selector of selectors) {
          const nodes = root.querySelectorAll(selector);
          nodes.forEach((node) => {
            if (!node) {
              return;
            }
            node.style.display = "none";
          });
        }
      }
    };
    suppressErrors();
    setInterval(suppressErrors, 800);
  }
  const submitMove = (coord) => {
    console.log(`[Gomoku] submitMove called with: ${coord}`);
    if (!coord) {
      return false;
    }
    const input = resolveControl("#__MOVE_INPUT_ID__", "textarea, input");
    const button = resolveControl("#__SUBMIT_BUTTON_ID__", "button");
    if (!input || !button) {
      console.warn("[Gomoku] Input or Button not found!", { input, button });
      window.__gomoku_pending_coord = coord;
      return false;
    }
    console.log("[Gomoku] Elements found. Setting value and dispatching.");

    // Robust value setter for Gradio 4 / Svelte / React
    const setNativeValue = (element, value) => {
      const desc = Object.getOwnPropertyDescriptor(element, 'value');
      const valueSetter = desc ? desc.set : null;
      const prototype = Object.getPrototypeOf(element);
      const protoDesc = Object.getOwnPropertyDescriptor(prototype, 'value');
      const prototypeValueSetter = protoDesc ? protoDesc.set : null;

      if (prototypeValueSetter && valueSetter !== prototypeValueSetter) {
        prototypeValueSetter.call(element, value);
      } else if (valueSetter) {
        valueSetter.call(element, value);
      } else {
        element.value = value;
      }
    };

    try {
        setNativeValue(input, coord);
    } catch (e) {
        console.warn("[Gomoku] specific setter failed, using direct assignment", e);
        input.value = coord;
    }

    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));

    // Small delay to ensure state update before click
    setTimeout(() => {
        button.click();
    }, 50);

    return true;
  };
  window.__gomoku_submit = submitMove;
  const flushPending = () => {
    if (!window.__gomoku_pending_coord) {
      return;
    }
    const coord = window.__gomoku_pending_coord;
    if (submitMove(coord)) {
      window.__gomoku_pending_coord = null;
    }
  };
"""
    helper_block = (
        helper_block.replace("__MOVE_INPUT_ID__", move_input_id).replace(
            "__SUBMIT_BUTTON_ID__", submit_button_id
        )
    )

    click_handler = ""
    if enable_click:
        click_handler = f"""
  if (!window.__gomoku_click_handler__) {{
    window.__gomoku_click_handler__ = true;
    const handleClick = (event) => {{
      let cell = event.target.closest && event.target.closest("[data-coord],[aria-label],[title]");
      if (!cell) {{
        const path = event.composedPath ? event.composedPath() : [];
        if (path.length) {{
          cell = path.find((node) => readCoord(node));
        }}
      }}
      if (!cell) {{
        return;
      }}
      const coord = readCoord(cell);
      console.log(`[Gomoku] Clicked cell: ${{coord}}`);
      if (!coord) {{
        return;
      }}
      if (!window.__gomoku_submit) {{
        console.error("[Gomoku] Submit function missing!");
        return;
      }}
      if (!window.__gomoku_submit(coord)) {{
        console.warn("[Gomoku] Submit returned false");
        return;
      }}
    }};
    const bindRoot = () => {{
      const root = getRoot();
      if (root && !root.__gomokuRootBound) {{
        root.__gomokuRootBound = true;
        root.addEventListener("click", handleClick, true);
      }}
      if (!document.__gomokuDocBound) {{
        document.__gomokuDocBound = true;
        document.addEventListener("click", handleClick, true);
      }}
    }};
    const bindBoard = () => {{
      const board = find("#{board_container_id}");
      if (!board) {{
        return;
      }}
      if (board.__gomokuBound) {{
        return;
      }}
      board.__gomokuBound = true;
      board.addEventListener("click", handleClick);
    }};
    const keepBinding = () => {{
      bindRoot();
      bindBoard();
      flushPending();
      setTimeout(keepBinding, 500);
    }};
    keepBinding();
  }}
"""

    refresh_handler = ""
    if refresh_button_id and refresh_interval_ms:
        refresh_handler = f"""
  if (!window.__gomoku_refresh_handler__) {{
    window.__gomoku_refresh_handler__ = true;
    const bindRefresh = () => {{
      const button = resolveControl("#{refresh_button_id}", "button");
      if (!button) {{
        setTimeout(bindRefresh, 300);
        return;
      }}
      if (window.__gomoku_refresh_interval__) {{
        return;
      }}
      button.click();
      window.__gomoku_refresh_interval__ = setInterval(() => {{
        button.click();
      }}, {int(refresh_interval_ms)});
    }};
    bindRefresh();
  }}
"""

    return f"""
(() => {{
{helper_block}{click_handler}{refresh_handler}
}})();
"""
