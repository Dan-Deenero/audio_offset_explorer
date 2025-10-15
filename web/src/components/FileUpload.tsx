"use client";

import React, { useRef, useEffect } from "react";

interface FileUploadProps {
  referenceFile: File | null;
  setReferenceFile: (file: File | null) => void;
  candidateFiles: File[];
  setCandidateFiles: (files: File[]) => void;
}

export default function FileUpload({
  referenceFile,
  setReferenceFile,
  candidateFiles,
  setCandidateFiles,
}: FileUploadProps) {
  const referenceInputRef = useRef<HTMLInputElement | null>(null);
  const candidatesInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // Cleanup object URLs when component unmounts
    return () => {
      if (referenceFile) URL.revokeObjectURL(URL.createObjectURL(referenceFile));
      candidateFiles.forEach((f) => URL.revokeObjectURL(URL.createObjectURL(f)));
    };
  }, [referenceFile, candidateFiles]);

  const onReferenceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setReferenceFile(file);
  };

  const onCandidatesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    setCandidateFiles(files);
  };

  return (
    <div className="flex flex-col md:flex-row gap-5">
      <div className="p-4 card shadow-sm bg-base-100 w-full overflow-hidden">
        <label className="block mb-2 card-title font-medium">
          Reference audio (single file)
        </label>
        <input
          ref={referenceInputRef}
          type="file"
          accept="audio/*"
          onChange={onReferenceChange}
          className="file-input file-input-ghost"
        />
        {referenceFile && (
          <div className="mt-2 flex card-title items-center gap-4">
            <div>{referenceFile.name}</div>
            <audio controls src={URL.createObjectURL(referenceFile)} />
          </div>
        )}
      </div>

      <div className="p-4 card shadow-sm bg-base-100 w-full overflow-hidden">
        <label className="block mb-2 font-medium">
          Candidate audio files (one or more)
        </label>
        <input
          ref={candidatesInputRef}
          type="file"
          accept="audio/*"
          multiple
          onChange={onCandidatesChange}
          className="file-input file-input-ghost"
        />
        {candidateFiles.length > 0 && (
          <div className="mt-2 space-y-1">
            <div className="text-sm text-gray-600">
              {candidateFiles.length} file(s) selected
            </div>
            <div className="flex gap-2 overflow-x-auto py-2">
              {candidateFiles.slice(0, 8).map((f) => (
                <div key={f.name} className="p-2 border rounded min-w-[120px]">
                  <div className="text-xs truncate">{f.name}</div>
                  <audio controls src={URL.createObjectURL(f)} />
                </div>
              ))}
            </div>
            {candidateFiles.length > 8 && (
              <div className="text-xs text-gray-500">...and more</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
