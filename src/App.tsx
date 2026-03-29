import React, { useState, useEffect, useRef } from 'react';
import * as faceapi from '@vladmandic/face-api';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import exifr from 'exifr';
import { GoogleGenAI } from '@google/genai';
import { Upload, User, Image as ImageIcon, Download, Loader2, Folder, X, AlertCircle, Search, StopCircle, Calendar, Maximize, FileText, Camera, Aperture, Timer, Wand2, FileSpreadsheet } from 'lucide-react';
import { cn } from './lib/utils';

type ImageMetadata = {
  url: string;
  filename: string;
  date: string | null;
  width: number;
  height: number;
  cameraModel?: string;
  aperture?: string;
  exposureTime?: string;
};

type FaceCluster = {
  id: string;
  name: string;
  descriptor: Float32Array;
  faceImage: string; // cropped face data URL
  sourceImages: ImageMetadata[]; // array of original image metadata
};

export default function App() {
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, currentFileName: '', facesFoundInCurrent: 0, totalFacesFound: 0 });
  const [clusters, setClusters] = useState<FaceCluster[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<FaceCluster | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImageMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [matchThreshold, setMatchThreshold] = useState(0.55);
  const [scoreThreshold, setScoreThreshold] = useState(0.5);
  const [iouThreshold, setIouThreshold] = useState(0.5);
  const [maxDetections, setMaxDetections] = useState(100);
  const [generatingAvatarId, setGeneratingAvatarId] = useState<string | null>(null);
  const cancelScanRef = useRef(false);

  const stopScan = () => {
    cancelScanRef.current = true;
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
        ]);
        setModelsLoaded(true);
      } catch (err) {
        console.error("Fejl ved indlæsning af modeller:", err);
        setError("Kunne ikke indlæse AI-modellerne. Tjek din internetforbindelse.");
      }
    };
    loadModels();
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsProcessing(true);
    setError(null);
    cancelScanRef.current = false;
    const currentClusters = [...clusters];
    let totalFaces = 0;

    for (let i = 0; i < files.length; i++) {
      if (cancelScanRef.current) {
        break;
      }
      const file = files[i];
      setProgress({ current: i + 1, total: files.length, currentFileName: file.name, facesFoundInCurrent: 0, totalFacesFound: totalFaces });
      
      // Kun billeder
      if (!file.type.startsWith('image/')) continue;

      const url = URL.createObjectURL(file);
      let usedUrl = false;
      let dateTaken = file.lastModified ? new Date(file.lastModified).toLocaleDateString() : null;
      let cameraModel, aperture, exposureTime;

      try {
        const exifData = await exifr.parse(file);
        if (exifData) {
          if (exifData.DateTimeOriginal) {
            dateTaken = new Date(exifData.DateTimeOriginal).toLocaleDateString();
          }
          if (exifData.Model) cameraModel = exifData.Model;
          if (exifData.FNumber) aperture = `f/${exifData.FNumber}`;
          if (exifData.ExposureTime) {
            exposureTime = exifData.ExposureTime >= 1 
              ? `${exifData.ExposureTime}s` 
              : `1/${Math.round(1 / exifData.ExposureTime)}s`;
          }
        }
      } catch (e) {
        // Ignorer EXIF fejl
      }

      try {
        const img = new Image();
        img.src = url;
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
        });

        const options = new faceapi.SsdMobilenetv1Options({ 
          minConfidence: scoreThreshold, 
          maxResults: maxDetections 
        });
        const detections = await faceapi.detectAllFaces(img, options).withFaceLandmarks().withFaceDescriptors();
        
        const imageMeta: ImageMetadata = {
          url,
          filename: file.name,
          date: dateTaken,
          width: img.width,
          height: img.height,
          cameraModel,
          aperture,
          exposureTime
        };

        totalFaces += detections.length;
        setProgress({ current: i + 1, total: files.length, currentFileName: file.name, facesFoundInCurrent: detections.length, totalFacesFound: totalFaces });

        for (const detection of detections) {
          let bestMatch = { distance: 1.0, cluster: null as FaceCluster | null };

          for (const cluster of currentClusters) {
            const distance = faceapi.euclideanDistance(detection.descriptor, cluster.descriptor);
            if (distance < bestMatch.distance) {
              bestMatch = { distance, cluster };
            }
          }

          if (bestMatch.distance < matchThreshold && bestMatch.cluster) {
            if (!bestMatch.cluster.sourceImages.some(img => img.url === url)) {
              bestMatch.cluster.sourceImages.push(imageMeta);
              usedUrl = true;
            }
          } else {
            // Opret ny klynge (person)
            const canvas = document.createElement('canvas');
            const box = detection.detection.box;
            
            // Tilføj lidt padding til ansigtet
            const padX = box.width * 0.2;
            const padY = box.height * 0.2;
            const sx = Math.max(0, box.x - padX);
            const sy = Math.max(0, box.y - padY);
            const sw = Math.min(img.width - sx, box.width + padX * 2);
            const sh = Math.min(img.height - sy, box.height + padY * 2);

            canvas.width = sw;
            canvas.height = sh;
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
              
              const newCluster: FaceCluster = {
                id: Math.random().toString(36).substring(7),
                name: '',
                descriptor: detection.descriptor,
                faceImage: canvas.toDataURL('image/jpeg', 0.8),
                sourceImages: [imageMeta]
              };
              currentClusters.push(newCluster);
              usedUrl = true;
            }
          }
        }
      } catch (err) {
        console.error("Fejl ved behandling af billede:", file.name, err);
      }
      
      if (!usedUrl) {
        URL.revokeObjectURL(url);
      }
      
      // Opdater state løbende så brugeren kan se fremskridt
      setClusters([...currentClusters]);
    }
    
    setIsProcessing(false);
    setProgress({ current: 0, total: 0, currentFileName: '', facesFoundInCurrent: 0, totalFacesFound: 0 });
    
    // Nulstil input
    event.target.value = '';
  };

  const handleFolderScan = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const namedClusters = clusters.filter(c => c.name.trim() !== '');
    if (namedClusters.length === 0) {
      setError("Du skal navngive mindst én person først, før du kan scanne en mappe for dem.");
      event.target.value = '';
      return;
    }

    setIsProcessing(true);
    setError(null);
    cancelScanRef.current = false;
    const currentClusters = [...clusters];
    let totalFaces = 0;

    for (let i = 0; i < files.length; i++) {
      if (cancelScanRef.current) {
        break;
      }
      const file = files[i];
      setProgress({ current: i + 1, total: files.length, currentFileName: file.name, facesFoundInCurrent: 0, totalFacesFound: totalFaces });
      
      if (!file.type.startsWith('image/')) continue;

      const url = URL.createObjectURL(file);
      let usedUrl = false;
      let dateTaken = file.lastModified ? new Date(file.lastModified).toLocaleDateString() : null;
      let cameraModel, aperture, exposureTime;

      try {
        const exifData = await exifr.parse(file);
        if (exifData) {
          if (exifData.DateTimeOriginal) {
            dateTaken = new Date(exifData.DateTimeOriginal).toLocaleDateString();
          }
          if (exifData.Model) cameraModel = exifData.Model;
          if (exifData.FNumber) aperture = `f/${exifData.FNumber}`;
          if (exifData.ExposureTime) {
            exposureTime = exifData.ExposureTime >= 1 
              ? `${exifData.ExposureTime}s` 
              : `1/${Math.round(1 / exifData.ExposureTime)}s`;
          }
        }
      } catch (e) {
        // Ignorer EXIF fejl
      }

      try {
        const img = new Image();
        img.src = url;
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
        });

        const options = new faceapi.SsdMobilenetv1Options({ 
          minConfidence: scoreThreshold, 
          maxResults: maxDetections 
        });
        const detections = await faceapi.detectAllFaces(img, options).withFaceLandmarks().withFaceDescriptors();
        
        const imageMeta: ImageMetadata = {
          url,
          filename: file.name,
          date: dateTaken,
          width: img.width,
          height: img.height,
          cameraModel,
          aperture,
          exposureTime
        };

        totalFaces += detections.length;
        setProgress({ current: i + 1, total: files.length, currentFileName: file.name, facesFoundInCurrent: detections.length, totalFacesFound: totalFaces });

        for (const detection of detections) {
          let bestMatch = { distance: 1.0, cluster: null as FaceCluster | null };

          for (const cluster of namedClusters) {
            const distance = faceapi.euclideanDistance(detection.descriptor, cluster.descriptor);
            if (distance < bestMatch.distance) {
              bestMatch = { distance, cluster };
            }
          }

          if (bestMatch.distance < matchThreshold && bestMatch.cluster) {
            const targetCluster = currentClusters.find(c => c.id === bestMatch.cluster!.id);
            if (targetCluster && !targetCluster.sourceImages.some(img => img.url === url)) {
              targetCluster.sourceImages.push(imageMeta);
              usedUrl = true;
            }
          }
        }
      } catch (err) {
        console.error("Fejl ved behandling af billede:", file.name, err);
      }
      
      if (!usedUrl) {
        URL.revokeObjectURL(url);
      }

      // Opdater state periodisk for at undgå at fryse UI ved store mapper
      if (i % 10 === 0) {
        setClusters([...currentClusters]);
      }
    }
    
    setClusters([...currentClusters]);
    setIsProcessing(false);
    setProgress({ current: 0, total: 0, currentFileName: '', facesFoundInCurrent: 0, totalFacesFound: 0 });
    event.target.value = '';
  };

  const updateClusterName = (id: string, newName: string) => {
    setClusters(clusters.map(c => c.id === id ? { ...c, name: newName } : c));
  };

  const downloadCluster = async (cluster: FaceCluster) => {
    try {
      const zip = new JSZip();
      const folderName = cluster.name.trim() || 'Ukendt_Person';
      const folder = zip.folder(folderName);
      
      if (!folder) throw new Error("Kunne ikke oprette mappe i ZIP");

      for (let i = 0; i < cluster.sourceImages.length; i++) {
        const url = cluster.sourceImages[i].url;
        const response = await fetch(url);
        const blob = await response.blob();
        folder.file(`billede_${i + 1}.jpg`, blob);
      }

      const content = await zip.generateAsync({ type: 'blob' });
      saveAs(content, `${folderName}.zip`);
    } catch (err) {
      console.error("Fejl ved download:", err);
      alert("Der opstod en fejl ved download af billederne.");
    }
  };

  const downloadAll = async () => {
    try {
      const zip = new JSZip();
      
      for (const cluster of clusters) {
        const folderName = cluster.name.trim() || `Ukendt_Person_${cluster.id}`;
        const folder = zip.folder(folderName);
        if (!folder) continue;

        for (let i = 0; i < cluster.sourceImages.length; i++) {
          const url = cluster.sourceImages[i].url;
          const response = await fetch(url);
          const blob = await response.blob();
          folder.file(`billede_${i + 1}.jpg`, blob);
        }
      }

      const content = await zip.generateAsync({ type: 'blob' });
      saveAs(content, `Alle_Ansigter.zip`);
    } catch (err) {
      console.error("Fejl ved download af alle:", err);
      alert("Der opstod en fejl ved download af alle billederne.");
    }
  };

  const exportClusterMetadataToCSV = (cluster: FaceCluster) => {
    try {
      const headers = ['Filnavn', 'Dato', 'Bredde', 'Højde', 'Kameramodel', 'Blænde', 'Eksponeringstid'];
      const rows = cluster.sourceImages.map(img => [
        `"${img.filename.replace(/"/g, '""')}"`,
        `"${img.date || ''}"`,
        img.width,
        img.height,
        `"${(img.cameraModel || '').replace(/"/g, '""')}"`,
        `"${(img.aperture || '').replace(/"/g, '""')}"`,
        `"${(img.exposureTime || '').replace(/"/g, '""')}"`
      ]);

      const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(','))
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const safeName = cluster.name.trim() ? cluster.name.trim().replace(/[^a-z0-9]/gi, '_').toLowerCase() : `ukendt_person_${cluster.id}`;
      saveAs(blob, `metadata_${safeName}.csv`);
    } catch (err) {
      console.error("Fejl ved eksport af CSV:", err);
      alert("Der opstod en fejl ved eksport af metadata.");
    }
  };

  const generateAvatar = async (cluster: FaceCluster) => {
    try {
      setGeneratingAvatarId(cluster.id);
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const prompt = cluster.name 
        ? `A clean, minimalist flat design avatar icon for a person named ${cluster.name}, solid pastel background, vector art style, high quality`
        : `A clean, minimalist flat design avatar icon for an anonymous person, solid pastel background, vector art style, high quality`;
        
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: {
          parts: [{ text: prompt }],
        },
      });
      
      const candidate = response.candidates?.[0];
      if (candidate?.content?.parts) {
        for (const part of candidate.content.parts) {
          if (part.inlineData) {
            const base64EncodeString = part.inlineData.data;
            const imageUrl = `data:${part.inlineData.mimeType || 'image/png'};base64,${base64EncodeString}`;
            
            setClusters(prev => prev.map(c => c.id === cluster.id ? { ...c, faceImage: imageUrl } : c));
            break;
          }
        }
      }
    } catch (err) {
      console.error("Fejl ved generering af avatar:", err);
      alert("Kunne ikke generere avatar. Prøv igen.");
    } finally {
      setGeneratingAvatarId(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      <header className="bg-white border-b border-gray-200 px-6 py-4 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg text-white">
              <User size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Lokal Ansigtsgenkendelse</h1>
              <p className="text-sm text-gray-500">Organiser dine billeder efter personer</p>
            </div>
          </div>
          
          {clusters.length > 0 && !isProcessing && (
            <button 
              onClick={downloadAll}
              className="flex items-center gap-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-lg transition-colors text-sm font-medium"
            >
              <Download size={16} />
              Download Alle Biblioteker (ZIP)
            </button>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-start gap-3">
            <AlertCircle className="shrink-0 mt-0.5" size={18} />
            <p>{error}</p>
          </div>
        )}

        {/* Upload Sektion */}
        <div className="mb-10">
          {!modelsLoaded ? (
            <div className="bg-white border-2 border-dashed border-gray-200 rounded-2xl p-12 flex flex-col items-center justify-center text-center">
              <Loader2 className="animate-spin text-blue-600 mb-4" size={40} />
              <h3 className="text-lg font-medium text-gray-900 mb-1">Indlæser AI-modeller...</h3>
              <p className="text-gray-500 max-w-md">
                Dette tager et øjeblik. Modellerne downloades til din browser, så al genkendelse kan ske lokalt og privat på din enhed.
              </p>
            </div>
          ) : isProcessing ? (
            <div className="bg-blue-50 border-2 border-dashed border-blue-300 rounded-2xl p-12 flex flex-col items-center justify-center text-center">
              <Loader2 className="animate-spin text-blue-600 mb-4" size={48} />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Analyserer billeder...</h3>
              <p className="text-gray-600 mb-1">Behandler billede {progress.current} af {progress.total}</p>
              <p className="text-sm text-gray-400 mb-4 truncate max-w-xs" title={progress.currentFileName}>
                {progress.currentFileName || 'Forbereder...'}
              </p>
              
              <div className="w-full max-w-md bg-gray-200 rounded-full h-2.5 overflow-hidden mb-6">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-out" 
                  style={{ width: `${(progress.current / progress.total) * 100}%` }}
                ></div>
              </div>

              <div className="flex gap-4 mb-8 text-sm w-full max-w-md justify-center">
                <div className="bg-white px-4 py-3 rounded-xl border border-gray-200 shadow-sm flex-1">
                  <span className="block text-gray-500 text-xs uppercase tracking-wider mb-1">Ansigter i billede</span>
                  <span className="block text-2xl font-bold text-gray-900">{progress.facesFoundInCurrent}</span>
                </div>
                <div className="bg-white px-4 py-3 rounded-xl border border-gray-200 shadow-sm flex-1">
                  <span className="block text-gray-500 text-xs uppercase tracking-wider mb-1">Ansigter i alt</span>
                  <span className="block text-2xl font-bold text-blue-600">{progress.totalFacesFound}</span>
                </div>
              </div>

              <button
                onClick={stopScan}
                className="flex items-center gap-2 bg-red-100 hover:bg-red-200 text-red-700 px-6 py-2 rounded-full font-medium transition-colors"
              >
                <StopCircle size={20} />
                Stop scanning
              </button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Option 1: Find nye personer */}
              <div className="bg-white border-2 border-dashed border-gray-300 hover:border-blue-500 hover:bg-blue-50/30 rounded-2xl p-8 text-center relative transition-colors group">
                <input 
                  type="file" 
                  multiple 
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" 
                />
                <div className="bg-blue-100 text-blue-600 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <User size={28} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">1. Opret Personer</h3>
                <p className="text-sm text-gray-500 max-w-xs mx-auto">
                  Upload et par referencebilleder for at finde ansigter og navngive personerne.
                </p>
              </div>

              {/* Option 2: Skan mappe for navngivne */}
              <div className="bg-white border-2 border-dashed border-gray-300 hover:border-green-500 hover:bg-green-50/30 rounded-2xl p-8 text-center relative transition-colors group">
                <input 
                  type="file" 
                  webkitdirectory="true"
                  multiple 
                  accept="image/*"
                  onChange={handleFolderScan}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" 
                />
                <div className="bg-green-100 text-green-600 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Search size={28} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">2. Skan Mappe</h3>
                <p className="text-sm text-gray-500 max-w-xs mx-auto">
                  Vælg en hel mappe. Vi finder og sorterer kun de personer, du allerede har navngivet.
                </p>
              </div>
            </div>
            
            <div className="mt-8 bg-white border border-gray-200 rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Indstillinger for Nøjagtighed</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium text-gray-700">Genkendelses-tolerance (Match)</label>
                    <span className="text-sm text-gray-500">{matchThreshold.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.40" max="0.70" step="0.01" 
                    value={matchThreshold} 
                    onChange={(e) => setMatchThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-gray-500 mt-2">Lavere værdi = strengere match (færre fejl, men kan misse ansigter). Højere værdi = løsere match.</p>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium text-gray-700">Score Threshold (Kvalitet)</label>
                    <span className="text-sm text-gray-500">{scoreThreshold.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.10" max="0.90" step="0.05" 
                    value={scoreThreshold} 
                    onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-gray-500 mt-2">Hvor sikker AI'en skal være på, at det er et ansigt. Lavere værdi = finder flere (også utydelige) ansigter.</p>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium text-gray-700">IoU Threshold (Overlap)</label>
                    <span className="text-sm text-gray-500">{iouThreshold.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.10" max="1.00" step="0.05" 
                    value={iouThreshold} 
                    onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-gray-500 mt-2">Kontrollerer hvor meget to ansigtsbokse må overlappe. Bruges til at fjerne dobbelte detektioner af samme ansigt.</p>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium text-gray-700">Max Detections (Antal)</label>
                    <span className="text-sm text-gray-500">{maxDetections}</span>
                  </div>
                  <input 
                    type="range" 
                    min="1" max="200" step="1" 
                    value={maxDetections} 
                    onChange={(e) => setMaxDetections(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-gray-500 mt-2">Det maksimale antal ansigter, der kan findes i ét enkelt billede. Sænk for bedre ydeevne på store gruppebilleder.</p>
                </div>
              </div>
            </div>
          </>
          )}
        </div>

        {/* Resultater */}
        {clusters.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <Folder className="text-gray-400" />
                Fundne Personer ({clusters.length})
              </h2>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {clusters.map(cluster => (
                <div key={cluster.id} className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow flex flex-col">
                  <div 
                    className="aspect-square bg-gray-100 relative cursor-pointer group"
                    onClick={() => {
                      setSelectedCluster(cluster);
                      setSelectedImage(cluster.sourceImages[0] || null);
                    }}
                  >
                    <img 
                      src={cluster.faceImage} 
                      alt="Beskåret ansigt" 
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                      <div className="opacity-0 group-hover:opacity-100 bg-white/90 text-gray-900 text-sm font-medium px-3 py-1.5 rounded-full shadow-sm transform translate-y-2 group-hover:translate-y-0 transition-all">
                        Se {cluster.sourceImages.length} billeder
                      </div>
                    </div>
                    <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm text-white text-xs font-medium px-2 py-1 rounded-md flex items-center gap-1">
                      <ImageIcon size={12} />
                      {cluster.sourceImages.length}
                    </div>
                  </div>
                  
                  <div className="p-4 flex-1 flex flex-col">
                    <div className="mb-4">
                      <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1 block">
                        Navngiv person
                      </label>
                      <input
                        type="text"
                        value={cluster.name}
                        onChange={(e) => updateClusterName(cluster.id, e.target.value)}
                        placeholder="F.eks. Jens Jensen"
                        className="w-full border-b-2 border-gray-200 focus:border-blue-600 outline-none py-1.5 text-gray-900 font-medium bg-transparent transition-colors placeholder:text-gray-400"
                      />
                    </div>
                    
                    <div className="mt-auto pt-2 flex gap-2">
                      <button
                        onClick={() => downloadCluster(cluster)}
                        className="flex-1 flex items-center justify-center gap-2 bg-gray-50 hover:bg-gray-100 text-gray-700 border border-gray-200 py-2 rounded-lg text-sm font-medium transition-colors"
                        title="Download Bibliotek"
                      >
                        <Download size={16} />
                      </button>
                      <button
                        onClick={() => generateAvatar(cluster)}
                        disabled={generatingAvatarId === cluster.id}
                        className="flex-1 flex items-center justify-center gap-2 bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                        title="Generer AI Avatar"
                      >
                        {generatingAvatarId === cluster.id ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
                        AI Avatar
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Modal til at se billeder af en person */}
      {selectedCluster && selectedImage && (
        <div className="fixed inset-0 bg-black/95 z-50 flex flex-col animate-in fade-in duration-200">
          <div className="flex items-center justify-between p-4 border-b border-white/10 bg-black/50 backdrop-blur-md shrink-0">
            <div className="flex items-center gap-4 text-white">
              <img src={selectedCluster.faceImage} className="w-10 h-10 rounded-full object-cover border border-white/20" />
              <div>
                <h3 className="font-medium text-lg">{selectedCluster.name || 'Ukendt Person'}</h3>
                <p className="text-sm text-gray-400">{selectedCluster.sourceImages.length} billeder fundet</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button 
                onClick={() => exportClusterMetadataToCSV(selectedCluster)}
                className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg transition-colors text-sm font-medium"
                title="Eksporter Metadata (CSV)"
              >
                <FileSpreadsheet size={16} />
                <span className="hidden sm:inline">CSV</span>
              </button>
              <button 
                onClick={() => downloadCluster(selectedCluster)}
                className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg transition-colors text-sm font-medium"
              >
                <Download size={16} />
                <span className="hidden sm:inline">Download</span>
              </button>
              <button 
                onClick={() => {
                  setSelectedCluster(null);
                  setSelectedImage(null);
                }}
                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-full transition-colors"
              >
                <X size={24} />
              </button>
            </div>
          </div>
          
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* Large Image Viewer */}
            <div className="flex-1 relative bg-black/80 flex items-center justify-center p-4 overflow-hidden">
              <img 
                src={selectedImage.url} 
                alt={selectedImage.filename} 
                className="max-w-full max-h-full object-contain drop-shadow-2xl"
              />
            </div>

            {/* Metadata Sidebar */}
            <div className="w-full lg:w-80 bg-gray-900 border-l border-white/10 p-6 overflow-y-auto shrink-0 flex flex-col gap-6">
              <h4 className="text-white font-medium text-lg border-b border-white/10 pb-3">Billedinfo</h4>
              
              <div className="flex flex-col gap-5">
                <div className="flex items-start gap-3 text-gray-300">
                  <FileText size={18} className="shrink-0 mt-0.5 text-gray-500" />
                  <div className="break-all">
                    <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Filnavn</p>
                    <p className="text-sm font-medium">{selectedImage.filename}</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3 text-gray-300">
                  <Maximize size={18} className="shrink-0 mt-0.5 text-gray-500" />
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Opløsning</p>
                    <p className="text-sm font-medium">{selectedImage.width} × {selectedImage.height} px</p>
                  </div>
                </div>
                
                {selectedImage.date && (
                  <div className="flex items-start gap-3 text-gray-300">
                    <Calendar size={18} className="shrink-0 mt-0.5 text-gray-500" />
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Dato taget</p>
                      <p className="text-sm font-medium">{selectedImage.date}</p>
                    </div>
                  </div>
                )}
                
                {selectedImage.cameraModel && (
                  <div className="flex items-start gap-3 text-gray-300">
                    <Camera size={18} className="shrink-0 mt-0.5 text-gray-500" />
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Kamera</p>
                      <p className="text-sm font-medium">{selectedImage.cameraModel}</p>
                    </div>
                  </div>
                )}
                
                {selectedImage.aperture && (
                  <div className="flex items-start gap-3 text-gray-300">
                    <Aperture size={18} className="shrink-0 mt-0.5 text-gray-500" />
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Blænde</p>
                      <p className="text-sm font-medium">{selectedImage.aperture}</p>
                    </div>
                  </div>
                )}

                {selectedImage.exposureTime && (
                  <div className="flex items-start gap-3 text-gray-300">
                    <Timer size={18} className="shrink-0 mt-0.5 text-gray-500" />
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Eksponeringstid</p>
                      <p className="text-sm font-medium">{selectedImage.exposureTime}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Thumbnails Strip */}
          <div className="h-28 bg-gray-950 border-t border-white/10 p-3 flex gap-3 overflow-x-auto shrink-0 items-center">
            {selectedCluster.sourceImages.map((meta, i) => (
              <button
                key={i}
                onClick={() => setSelectedImage(meta)}
                className={cn(
                  "h-full aspect-square shrink-0 rounded-lg overflow-hidden border-2 transition-all",
                  selectedImage === meta ? "border-blue-500 opacity-100 scale-105" : "border-transparent opacity-50 hover:opacity-100 hover:scale-105"
                )}
              >
                <img src={meta.url} alt={meta.filename} className="w-full h-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
