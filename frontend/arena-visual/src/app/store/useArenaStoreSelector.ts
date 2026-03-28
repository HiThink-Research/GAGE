import { useSyncExternalStore } from "react";

interface ExternalStore<TSnapshot> {
  getSnapshot: () => TSnapshot;
  subscribe: (listener: () => void) => () => void;
}

export function useArenaStoreSelector<TSnapshot, TSelected>(
  store: ExternalStore<TSnapshot>,
  selector: (snapshot: TSnapshot) => TSelected,
): TSelected {
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.getSnapshot()),
    () => selector(store.getSnapshot()),
  );
}
