"use client"

import { useState, useEffect } from "react"

interface Detection {
  id: string
  timestamp: string
  type: string
  classification: "safe" | "suspicious" | "malicious"
  confidence: number
  explanation: string
}

const mockDetections: Detection[] = [
  {
    id: "1",
    timestamp: "2024-12-26 14:32:15",
    type: "email",
    classification: "malicious",
    confidence: 0.92,
    explanation: "Suspicious content detected | Domain spoofing detected",
  },
  {
    id: "2",
    timestamp: "2024-12-26 14:15:42",
    type: "url",
    classification: "suspicious",
    confidence: 0.65,
    explanation: "Shortened URL detected | Suspicious redirect",
  },
  {
    id: "3",
    timestamp: "2024-12-26 13:48:09",
    type: "email",
    classification: "safe",
    confidence: 0.15,
    explanation: "Content matches safe threat pattern",
  },
  {
    id: "4",
    timestamp: "2024-12-26 13:22:11",
    type: "sms",
    classification: "malicious",
    confidence: 0.88,
    explanation: "Urgent language detected | Shortened URL detected",
  },
]

const classificationStyles = {
  safe: "bg-safe/10 text-safe border-safe/30",
  suspicious: "bg-suspicious/10 text-suspicious border-suspicious/30",
  malicious: "bg-malicious/10 text-malicious border-malicious/30",
}

export default function RecentDetections() {
  const [detections, setDetections] = useState<Detection[]>([])

  useEffect(() => {
    // Simulate fetching detections from API
    setDetections(mockDetections)
  }, [])

  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h3 className="text-lg font-bold text-foreground mb-4">Recent Detections</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-3 px-4 text-foreground/70 font-medium">Time</th>
              <th className="text-left py-3 px-4 text-foreground/70 font-medium">Type</th>
              <th className="text-left py-3 px-4 text-foreground/70 font-medium">Classification</th>
              <th className="text-left py-3 px-4 text-foreground/70 font-medium">Confidence</th>
              <th className="text-left py-3 px-4 text-foreground/70 font-medium">Details</th>
            </tr>
          </thead>
          <tbody>
            {detections.map((detection) => (
              <tr key={detection.id} className="border-b border-border hover:bg-border/30 transition">
                <td className="py-3 px-4 text-foreground/70">{detection.timestamp}</td>
                <td className="py-3 px-4 text-foreground capitalize">{detection.type}</td>
                <td className="py-3 px-4">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-semibold border ${classificationStyles[detection.classification]}`}
                  >
                    {detection.classification.charAt(0).toUpperCase() + detection.classification.slice(1)}
                  </span>
                </td>
                <td className="py-3 px-4 text-foreground">{(detection.confidence * 100).toFixed(0)}%</td>
                <td className="py-3 px-4 text-foreground/70 text-xs">{detection.explanation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
