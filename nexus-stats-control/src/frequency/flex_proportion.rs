/// Global event counter with decay for proportion tracking.
///
/// Users create one `FlexProportionGlobal` and multiple `FlexProportionEntity`
/// instances. Each entity tracks its share of the global total with temporal
/// decay — old activity fades, recent activity dominates.
///
/// # Usage
///
/// The user is responsible for calling `global.update()` once per event.
/// Entities take the current period as a plain `u64` — no mutable reference
/// to the global is needed.
///
/// ```ignore
/// let mut global = FlexProportionGlobal::new(1000);
/// let mut entity_a = FlexProportionEntity::new();
///
/// // Record an event for entity A
/// global.update();
/// entity_a.update(global.period());
///
/// // Query fraction
/// let frac = entity_a.fraction(global.total(), global.period());
/// ```
///
/// # Use Cases
/// - "What fraction of total traffic goes to each venue?"
/// - Fair-share scheduling input
/// - Dynamic load distribution tracking
#[derive(Debug, Clone)]
pub struct FlexProportionGlobal {
    total: u64,
    half_life: u64,
    period: u64,
}

/// Per-entity event counter for proportion tracking.
///
/// Decoupled from the global tracker — takes plain values instead of
/// references. The user calls `global.update()` separately.
#[derive(Debug, Clone)]
pub struct FlexProportionEntity {
    count: u64,
    period: u64,
}

impl FlexProportionGlobal {
    /// Creates a new global tracker.
    ///
    /// `half_life_events` is the number of global events after which old
    /// contributions decay by half.
    #[inline]
    pub fn new(half_life_events: u64) -> Result<Self, nexus_stats_core::ConfigError> {
        if half_life_events == 0 {
            return Err(nexus_stats_core::ConfigError::Invalid(
                "half_life_events must be positive",
            ));
        }
        Ok(Self {
            total: 0,
            half_life: half_life_events,
            period: 0,
        })
    }

    /// Updates the global event count. Call this once per event, before
    /// updating the entity.
    #[inline]
    pub fn update(&mut self) {
        self.total += 1;
        if self.total.is_multiple_of(self.half_life) {
            self.period += 1;
        }
    }

    /// Total global events recorded.
    #[inline]
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Current decay period.
    #[inline]
    #[must_use]
    pub fn period(&self) -> u64 {
        self.period
    }
}

impl FlexProportionEntity {
    /// Creates a new entity tracker.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            period: 0,
        }
    }

    /// Records an event for this entity.
    ///
    /// Pass the current global period (from `global.period()`). The entity
    /// applies decay catch-up if the period has advanced, then increments
    /// its count.
    ///
    /// **Important:** Call `global.update()` separately — this method does
    /// NOT update the global tracker.
    #[inline]
    pub fn update(&mut self, current_period: u64) {
        while self.period < current_period {
            self.count /= 2;
            self.period += 1;
        }
        self.count += 1;
    }

    /// Fraction of global total attributed to this entity (0.0 to 1.0).
    ///
    /// Pass the current global total and period. Returns 0.0 if total is zero.
    #[inline]
    #[must_use]
    pub fn fraction(&self, total: u64, current_period: u64) -> f64 {
        if total == 0 {
            return 0.0;
        }

        // Decay count to current period
        let mut count = self.count;
        let mut period = self.period;
        while period < current_period {
            count /= 2;
            period += 1;
        }

        count as f64 / total as f64
    }

    /// This entity's current (possibly decayed) event count.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Resets this entity's count.
    #[inline]
    pub fn reset(&mut self) {
        self.count = 0;
        self.period = 0;
    }
}

impl Default for FlexProportionEntity {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_entity_full_share() {
        let mut global = FlexProportionGlobal::new(100).unwrap();
        let mut entity = FlexProportionEntity::new();

        for _ in 0..50 {
            global.update();
            entity.update(global.period());
        }

        let frac = entity.fraction(global.total(), global.period());
        assert!(frac > 0.0, "single entity should have positive fraction");
    }

    #[test]
    fn equal_entities_equal_share() {
        let mut global = FlexProportionGlobal::new(1000).unwrap();
        let mut e1 = FlexProportionEntity::new();
        let mut e2 = FlexProportionEntity::new();

        for _ in 0..100 {
            global.update();
            e1.update(global.period());
            global.update();
            e2.update(global.period());
        }

        let f1 = e1.fraction(global.total(), global.period());
        let f2 = e2.fraction(global.total(), global.period());
        assert!(
            (f1 - f2).abs() < 0.1,
            "equal entities should have equal fraction: {f1} vs {f2}"
        );
    }

    #[test]
    fn new_entity_ramps_up() {
        let mut global = FlexProportionGlobal::new(100).unwrap();
        let mut old = FlexProportionEntity::new();

        for _ in 0..50 {
            global.update();
            old.update(global.period());
        }

        let mut new = FlexProportionEntity::new();
        for _ in 0..10 {
            global.update();
            new.update(global.period());
        }

        let f_new = new.fraction(global.total(), global.period());
        assert!(f_new > 0.0, "new entity should have some fraction");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn empty_global() {
        let global = FlexProportionGlobal::new(100).unwrap();
        let entity = FlexProportionEntity::new();
        assert_eq!(entity.fraction(global.total(), global.period()), 0.0);
    }

    #[test]
    fn reset_entity() {
        let mut global = FlexProportionGlobal::new(100).unwrap();
        let mut entity = FlexProportionEntity::new();

        for _ in 0..20 {
            global.update();
            entity.update(global.period());
        }
        entity.reset();
        assert_eq!(entity.count(), 0);
    }

    #[test]
    fn default_entity() {
        let entity = FlexProportionEntity::default();
        assert_eq!(entity.count(), 0);
    }

    #[test]
    fn rejects_zero_half_life() {
        assert!(matches!(
            FlexProportionGlobal::new(0),
            Err(nexus_stats_core::ConfigError::Invalid(_))
        ));
    }
}
