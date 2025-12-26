"use client"

import { useState } from "react"
import DetectionResultComponent from "./detection-result"

interface DetectionResult {
  classification: "safe" | "suspicious" | "malicious"
  confidence: number
  explanation: string
  risk_level: string
  features: Record<string, any>
}

export default function DetectionForm() {
  const [activeTab, setActiveTab] = useState("email")
  const [emailBody, setEmailBody] = useState("")
  const [sender, setSender] = useState("")
  const [url, setUrl] = useState("")
  const [smsText, setSmsText] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)

  const handleDetect = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      const mockResult: DetectionResult = {
        classification: "suspicious",
        confidence: 0.65,
        explanation: "Suspicious content detected | Shortened URL detected",
        risk_level: "medium",
        features: {
          suspicious_content: 0.5,
          sender_reputation: 0.3,
        },
      }
      setResult(mockResult)
      setLoading(false)
    }, 1000)
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Analyze Content</h1>
        <p className="text-border">Detect phishing threats in emails, URLs, and SMS</p>
      </div>

      <div className="bg-card border border-border rounded-xl">
        {/* Tabs */}
        <div className="border-b border-border flex">
          {["email", "url", "sms"].map((tab) => (
            <button
              key={tab}
              onClick={() => {
                setActiveTab(tab)
                setResult(null)
              }}
              className={`flex-1 py-4 font-medium text-center transition ${
                activeTab === tab
                  ? "border-b-2 border-primary text-primary"
                  : "text-foreground/50 hover:text-foreground"
              }`}
            >
              {tab.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {activeTab === "email" && (
            <>
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">Sender Email</label>
                <input
                  type="email"
                  value={sender}
                  onChange={(e) => setSender(e.target.value)}
                  placeholder="sender@example.com"
                  className="w-full bg-border/30 border border-border rounded-lg px-4 py-2 text-foreground placeholder-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">Email Body</label>
                <textarea
                  value={emailBody}
                  onChange={(e) => setEmailBody(e.target.value)}
                  placeholder="Paste the email content here..."
                  rows={8}
                  className="w-full bg-border/30 border border-border rounded-lg px-4 py-2 text-foreground placeholder-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                />
              </div>
            </>
          )}

          {activeTab === "url" && (
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">URL to Analyze</label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                className="w-full bg-border/30 border border-border rounded-lg px-4 py-2 text-foreground placeholder-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          )}

          {activeTab === "sms" && (
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">SMS Text</label>
              <textarea
                value={smsText}
                onChange={(e) => setSmsText(e.target.value)}
                placeholder="Paste the SMS message here..."
                rows={4}
                className="w-full bg-border/30 border border-border rounded-lg px-4 py-2 text-foreground placeholder-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              />
            </div>
          )}

          <button
            onClick={handleDetect}
            disabled={loading}
            className="w-full bg-gradient-to-r from-primary to-primary-dark hover:opacity-90 disabled:opacity-50 text-white font-semibold py-3 rounded-lg transition"
          >
            {loading ? "Analyzing..." : "Analyze for Threats"}
          </button>
        </div>
      </div>

      {result && <DetectionResultComponent result={result} />}
    </div>
  )
}
