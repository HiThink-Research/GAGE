import { useCallback, useRef, useSyncExternalStore } from "react";

interface ExternalStore<TSnapshot> {
  getSnapshot: () => TSnapshot;
  subscribe: (listener: () => void) => () => void;
}

export function useArenaStoreSelector<TSnapshot, TSelected>(
  store: ExternalStore<TSnapshot>,
  selector: (snapshot: TSnapshot) => TSelected,
  isEqual: (left: TSelected, right: TSelected) => boolean = Object.is,
): TSelected {
  const selectorRef = useRef(selector);
  const isEqualRef = useRef(isEqual);
  const cachedSelectionRef = useRef<{
    hasValue: boolean;
    value: TSelected;
  }>({
    hasValue: false,
    value: undefined as TSelected,
  });

  selectorRef.current = selector;
  isEqualRef.current = isEqual;

  const getSelectedSnapshot = useCallback(() => {
    const nextSelection = selectorRef.current(store.getSnapshot());
    const cachedSelection = cachedSelectionRef.current;
    if (
      cachedSelection.hasValue &&
      isEqualRef.current(cachedSelection.value, nextSelection)
    ) {
      return cachedSelection.value;
    }

    cachedSelectionRef.current = {
      hasValue: true,
      value: nextSelection,
    };
    return nextSelection;
  }, [store]);

  return useSyncExternalStore(
    store.subscribe,
    getSelectedSnapshot,
    getSelectedSnapshot,
  );
}
