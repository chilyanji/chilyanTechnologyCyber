interface StatCardProps {
  label: string
  value: number | string
  subtext?: string
  icon?: string
  color?: "primary" | "safe" | "suspicious" | "malicious"
}

const colorMap = {
  primary: "from-primary to-primary-dark",
  safe: "from-safe/20 to-safe/10",
  suspicious: "from-suspicious/20 to-suspicious/10",
  malicious: "from-malicious/20 to-malicious/10",
}

const textColorMap = {
  primary: "text-primary",
  safe: "text-safe",
  suspicious: "text-suspicious",
  malicious: "text-malicious",
}

export default function StatCard({ label, value, subtext, icon, color = "primary" }: StatCardProps) {
  return (
    <div className={`bg-gradient-to-br ${colorMap[color]} border border-border rounded-xl p-6`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-foreground/70 text-sm font-medium mb-2">{label}</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-3xl font-bold text-foreground">
              {typeof value === "number" ? value.toLocaleString() : value}
            </h3>
            {subtext && <span className={`text-sm font-semibold ${textColorMap[color]}`}>{subtext}</span>}
          </div>
        </div>
        {icon && <span className="text-3xl">{icon}</span>}
      </div>
    </div>
  )
}
