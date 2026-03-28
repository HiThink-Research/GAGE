export type ArenaAction = Record<string, unknown>;

export interface ActionIntent<TAction extends ArenaAction = ArenaAction>
  extends Record<string, unknown> {
  playerId: string;
  action: TAction;
}

export interface InputInterpreter<
  TDeviceEvent,
  TIntent extends ActionIntent = ActionIntent,
> {
  interpret(event: TDeviceEvent): TIntent;
}

export function createInputInterpreter<
  TDeviceEvent,
  TIntent extends ActionIntent = ActionIntent,
>(interpret: (event: TDeviceEvent) => TIntent): InputInterpreter<TDeviceEvent, TIntent> {
  return { interpret };
}
