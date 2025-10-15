import { NextResponse } from "next/server";
import { stopProcess } from "../processManager";

export async function POST() {
  const stopped = stopProcess();
  if (stopped) {
    return NextResponse.json({ message: "Process stopped successfully." });
  }
  return NextResponse.json({ message: "No active process to stop." });
}
