import { NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function GET() {
  const encoder = new TextEncoder();z

  const stream = new ReadableStream({
    start(controller) {
      const backendRoot = path.join(process.cwd(), "../backend");
      const pythonScript = path.join(backendRoot, "match.py");

      // 👉 For streaming only logs (no file handling here)
      const python = spawn("python", [pythonScript, "--help"]);

      python.stdout.on("data", (data) => {
        controller.enqueue(encoder.encode(`data: ${data.toString()}\n\n`));
      });

      python.stderr.on("data", (data) => {
        controller.enqueue(encoder.encode(`data: ⚠️ ${data.toString()}\n\n`));
      });

      python.on("close", (code) => {
        controller.enqueue(encoder.encode(`data: ✅ Process finished (code ${code})\n\n`));
        controller.close();
      });

      python.on("error", (err) => {
        controller.enqueue(encoder.encode(`data: ❌ Error: ${err.message}\n\n`));
        controller.close();
      });
    },
  });

  return new NextResponse(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
