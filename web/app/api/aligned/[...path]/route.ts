import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";

export async function GET(
  req: Request,
  context: { params: Promise<{ path: string[] }> }  // 👈 await required
) {
  const { path: pathSegments } = await context.params; // 👈 fixed
  const relativePath = pathSegments.join(path.sep);

  const filePath = path.join(
    process.cwd(),
    "../backend/processed",
    relativePath
  );

  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: "File not found" }, { status: 404 });
  }

  const fileBuffer = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  let contentType = "application/octet-stream";

  if (ext === ".wav") contentType = "audio/wav";
  if (ext === ".mp3") contentType = "audio/mpeg";

  return new NextResponse(fileBuffer, {
    headers: {
      "Content-Type": contentType,
      "Content-Disposition": `inline; filename="${path.basename(filePath)}"`,
    },
  });
}
