"use server";

import { NextResponse } from "next/server";
import path from "path";
import { spawn } from "child_process";
import fs from "fs/promises";
import { mkdirSync, existsSync } from "fs";

// Helper: delete directory safely
async function cleanupDir(dirPath: string) {
  try {
    if (existsSync(dirPath)) {
      await fs.rm(dirPath, { recursive: true, force: true });
    }
  } catch (err) {
    console.warn(`⚠️ Cleanup failed for ${dirPath}:`, err);
  }
}

async function findReportFile(dir: string): Promise<string | null> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isFile() && entry.name.startsWith("offsets_") && entry.name.endsWith(".json")) {
      return fullPath;
    }
    if (entry.isDirectory()) {
      const found = await findReportFile(fullPath);
      if (found) return found;
    }
  }
  return null;
}

// POST handler for file upload and analysis
export async function POST(req: Request) {
  const controller = new AbortController();
  const timeoutMs = 10 * 60 * 1000; // 4 minutes
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const formData = await req.formData();
    const referenceFile = formData.get("reference") as File | null;
    const candidateFiles = formData.getAll("candidates") as File[];

    if (!referenceFile || candidateFiles.length === 0) {
      return NextResponse.json(
        { error: "Reference and candidate audio files are required." },
        { status: 400 }
      );
    }

    const backendRoot = path.join(process.cwd(), "../backend");
    const referenceDir = path.join(backendRoot, "uploads", "reference");
    const candidatesDir = path.join(backendRoot, "uploads", "candidates");
    const outputDir = path.join(backendRoot, "processed");

    [referenceDir, candidatesDir, outputDir].forEach((dir) =>
      mkdirSync(dir, { recursive: true })
    );

    const liveLogs: string[] = [];

    // Save reference file
    const refPath = path.join(referenceDir, referenceFile.name);
    const refBuffer = Buffer.from(await referenceFile.arrayBuffer());
    await fs.writeFile(refPath, refBuffer);
    liveLogs.push(`✅ Reference uploaded: ${referenceFile.name}`);
    console.log(`✅ Reference uploaded: ${referenceFile.name}`);

    // Save candidate files
    for (const file of candidateFiles) {
      const filePath = path.join(candidatesDir, file.name);
      const buffer = Buffer.from(await file.arrayBuffer());
      await fs.writeFile(filePath, buffer);
      liveLogs.push(`✅ Candidate uploaded: ${file.name}`);
      console.log(`✅ Candidate uploaded: ${file.name}`);
    }

    // Run Python script
    const pythonScript = path.join(backendRoot, "match.py");
    const args = [
      pythonScript,
      "--audio-dir",
      candidatesDir,
      "--unified",
      refPath,
      "-o",
      outputDir,
    ];

    console.log("▶️ Running Python CLI:", ["python", ...args].join(" "));

    const python = spawn("python", args, { signal: controller.signal });

    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data) => {
      const msg = data.toString();
      stdout += msg;
      liveLogs.push(msg.trim());
    });

    python.stderr.on("data", (data) => {
      const msg = data.toString();
      stderr += msg;
      liveLogs.push(`⚠️ ${msg.trim()}`);
    });

    const exitCode: number = await new Promise((resolve) => {
      python.on("close", resolve);
    });

    clearTimeout(timeout);

    if (controller.signal.aborted) {
      python.kill("SIGKILL");
      await cleanupDir(candidatesDir);
      await cleanupDir(referenceDir);
      return NextResponse.json(
        {
          error: "Processing timed out after 4 minutes.",
          logs: liveLogs.slice(-20),
        },
        { status: 504 }
      );
    }

    if (exitCode !== 0) {
      await cleanupDir(candidatesDir);
      await cleanupDir(referenceDir);
      return NextResponse.json(
        {
          error: "Python script failed.",
          details: stderr || stdout,
          logs: liveLogs.slice(-20),
        },
        { status: 500 }
      );
    }

    // Read and parse report
    const files = await fs.readdir(outputDir);
    const reportFile = files.find(
      (f) => f.startsWith("offsets_") && f.endsWith(".json")
    );

    const reportPath = await findReportFile(outputDir);

    if (!reportPath) {
      console.error("❌ Report file not found in:", outputDir);
      return NextResponse.json(
        {
          error: "Python completed but no report file found.",
          logs: liveLogs.slice(-20),
        },
        { status: 500 }
      );
    }

    let reportData: any;
    try {
      const rawData = await fs.readFile(reportPath, "utf-8");
      reportData = JSON.parse(rawData);
    } catch (parseErr) {
      console.error("❌ Failed to parse report file:", parseErr);
      return NextResponse.json(
        {
          error: "Report file could not be parsed.",
          logs: liveLogs.slice(-20),
        },
        { status: 500 }
      );
    }


    if (!reportData || !Array.isArray(reportData.items)) {
      console.error("❌ Report file missing expected 'items' array");
      return NextResponse.json(
        {
          error: "Report file is malformed or contains no results.",
          logs: liveLogs.slice(-20),
        },
        { status: 500 }
      );
    }

    if (reportData.items.length === 0) {
      console.warn("⚠️ Report contains 0 matched items.");
    }

    const results = (reportData.items || []).map((item: any) => ({
      filename: item.video_file,
      decision: item.decision,
      offset: item.final_offset_seconds ?? 0,
      confidence: item.confidence ?? 0,
      sanity_warning: item.sanity_warning === false ? null : item.sanity_warning,
      raw_offset: item.signed_offset_seconds ?? 0,
      tags: item.metadata?.tags || [],
    }));

    console.log("✅ Parsed results:", results.length);


    // Cleanup uploads
    await cleanupDir(candidatesDir);
    await cleanupDir(referenceDir);

    return NextResponse.json({
      reference: path.basename(refPath),
      generatedAt: reportData.generated_at,
      total: results.length,
      results,
      logs: liveLogs.slice(-10),
    });

  } catch (err: any) {
    clearTimeout(timeout);
    console.error("❌ Backend error:", err);
    return NextResponse.json(
      {
        error:
          err.name === "AbortError"
            ? "Process terminated due to timeout."
            : err.message || "Internal server error.",
      },
      { status: 500 }
    );
  }
}
