import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";

export async function GET(
  req: Request,
  context: { params: Promise<{ path: string[] }> }  
) {

  // Extract the path segments from the dynamic route
  //    e.g. ["T1_Krept___Konan_Set_trimmed", "file_aligned.wav"]
  const { path: pathSegments } = await context.params; // 👈 fixed

  // Join path segments to reconstruct the relative path to the file
  const relativePath = pathSegments.join(path.sep);

  
  // Build the absolute file path pointing to backend/processed
  const filePath = path.join(
    process.cwd(),
    "../backend/processed",
    relativePath
  );

   //  Handle case where file does not exist
  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: "File not found" }, { status: 404 });
  }

  const fileBuffer = fs.readFileSync(filePath);

  // Determine the appropriate content type based on file extension
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
