"use client";

import React, { useRef, useState, useEffect } from "react";
import { CheckCircle, Music, X } from "lucide-react";

interface FileUploadProps {
    referenceFile: File | null;
    setReferenceFile: (file: File | null) => void;
    candidateFiles: File[];
    setCandidateFiles: (files: File[]) => void;
}


interface FileProgress {
    name: string;
    progress: number;
    status: "uploading" | "completed";
}

export default function FileUpload({
    referenceFile,
    setReferenceFile,
    candidateFiles,
    setCandidateFiles,
}: FileUploadProps) {
    const referenceInputRef = useRef<HTMLInputElement | null>(null);
    const candidatesInputRef = useRef<HTMLInputElement | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [fileProgress, setFileProgress] = useState<Record<string, FileProgress>>({});



    useEffect(() => {
        if (!referenceFile && referenceInputRef.current) {
            referenceInputRef.current.value = "";  // ✅ clears the file input
        }
    }, [referenceFile]);

    useEffect(() => {
        if (candidateFiles.length === 0 && candidatesInputRef.current) {
            candidatesInputRef.current.value = "";
        }
    }, [candidateFiles]);



    // 📌 simulate upload progress
    const simulateUpload = (file: File) => {
        setFileProgress((prev) => ({
            ...prev,
            [file.name]: { name: file.name, progress: 0, status: "uploading" },
        }));

        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.floor(Math.random() * 15) + 10; // random increment
            if (progress >= 100) {
                clearInterval(interval);
                setFileProgress((prev) => ({
                    ...prev,
                    [file.name]: { ...prev[file.name], progress: 100, status: "completed" },
                }));
            } else {
                setFileProgress((prev) => ({
                    ...prev,
                    [file.name]: { ...prev[file.name], progress, status: "uploading" },
                }));
            }
        }, 300);
    };

    const handleReference = (file: File | null) => {
        setReferenceFile(file);
        if (file) simulateUpload(file);
    };

    const handleCandidates = (files: File[]) => {
        setCandidateFiles(files);
        files.forEach(simulateUpload);
    };

    const onReferenceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] ?? null;
        handleReference(file);
    };

    const onCandidatesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files ? Array.from(e.target.files) : [];
        handleCandidates(files);
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>, type: "reference" | "candidates") => {
        e.preventDefault();
        setDragActive(false);
        const files = Array.from(e.dataTransfer.files);
        if (type === "reference" && files[0]) handleReference(files[0]);
        if (type === "candidates") handleCandidates(files);
    };

    const removeCandidateFile = (name: string) => {
        setCandidateFiles(candidateFiles.filter((f) => f.name !== name));
        setFileProgress((prev) => {
            const updated = { ...prev };
            delete updated[name];
            return updated;
        });
    };

    return (
        <div className="flex flex-col md:flex-row gap-5">
            {/* Reference Upload */}
            <div
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => handleDrop(e, "reference")}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${dragActive ? "border-primary bg-base-200" : "border-gray-300 bg-base-100"
                    } w-full`}
                onClick={() => referenceInputRef.current?.click()}
            >
                <input
                    ref={referenceInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={onReferenceChange}
                    className="hidden"
                />
                <div className="flex flex-col items-center gap-2">
                    <Music className="w-10 h-10 text-gray-500" />
                    <p className="text-sm text-gray-600">
                        {referenceFile ? (
                            <span className="font-medium">{referenceFile.name}</span>
                        ) : (
                            <>Drag & drop reference audio or <span className="text-primary underline">browse</span></>
                        )}
                    </p>
                </div>
                {referenceFile && (
                    <div className="mt-4 space-y-2 text-left">
                        <div className="flex justify-between items-center text-sm">
                            <span className="truncate">{referenceFile.name}</span>
                            {fileProgress[referenceFile.name]?.status === "completed" && (
                                <CheckCircle className="text-primary w-5 h-5" />
                            )}
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                            <div
                                className={`h-2 ${fileProgress[referenceFile.name]?.status === "completed" ? "bg-primary" : "bg-info"
                                    }`}
                                style={{ width: `${fileProgress[referenceFile.name]?.progress || 0}%` }}
                            />
                        </div>
                        <audio controls src={URL.createObjectURL(referenceFile)} className="w-full mt-2" />
                    </div>
                )}
            </div>

            {/* Candidate Upload */}
            <div
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => handleDrop(e, "candidates")}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${dragActive ? "border-primary bg-base-200" : "border-gray-300 bg-base-100"
                    } w-full`}
                onClick={() => candidatesInputRef.current?.click()}
            >
                <input
                    ref={candidatesInputRef}
                    type="file"
                    accept="audio/*"
                    multiple
                    onChange={onCandidatesChange}
                    className="hidden"
                />
                <div className="flex flex-col items-center gap-2">
                    <Music className="w-10 h-10 text-gray-500" />
                    <p className="text-sm text-gray-600">
                        {candidateFiles.length > 0 ? (
                            <span className="font-medium">{candidateFiles.length} file(s) selected</span>
                        ) : (
                            <>Drag & drop candidate audios or <span className="text-primary underline">browse</span></>
                        )}
                    </p>
                </div>

                {candidateFiles.length > 0 && (
                    <div className="mt-4 grid grid-cols-2 gap-2 space-y-2 max-h-60 overflow-y-auto">
                        {candidateFiles.map((file) => (
                            <div
                                key={file.name}
                                className="flex flex-col rounded-lg p-2 bg-base-100 shadow-md"
                            >
                                <div className="flex justify-between items-center gap-2">
                                    <div className="flex items-center gap-2 truncate">
                                        <Music className="w-8 h-8 text-blue-500" />
                                        <span className="truncate text-sm">{file.name}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {fileProgress[file.name]?.status === "completed" && (
                                            <CheckCircle className="text-primary w-5 h-5" />
                                        )}
                                        <button
                                            onClick={(e) => { e.stopPropagation(); removeCandidateFile(file.name); }}
                                            className="text-gray-400 hover:text-red-500"
                                        >
                                            <X className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2 mt-2 overflow-hidden">
                                    <div
                                        className={`h-2 ${fileProgress[file.name]?.status === "completed" ? "bg-primary" : "bg-info"
                                            }`}
                                        style={{ width: `${fileProgress[file.name]?.progress || 0}%` }}
                                    />
                                </div>
                                <audio controls src={URL.createObjectURL(file)} className="w-full mt-2 h-10" />
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
