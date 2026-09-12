function SectionedView({ componentBySection, section, presetTicker, onNavigateToTicker, region }) {
  const ActiveComponent = componentBySection[section] ?? Object.values(componentBySection)[0]

  return (
    <div className="dashboard">
      <section className="panel">
        <ActiveComponent presetTicker={presetTicker} onNavigateToTicker={onNavigateToTicker} region={region} />
      </section>
    </div>
  )
}

export default SectionedView
