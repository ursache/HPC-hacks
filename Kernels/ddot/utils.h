  #include <linux/kernel.h>

/* All counters, including PMCCNTR_EL0, are disabled/enabled */

#define QUADD_ARMV8_PMCR_E      (1 << 0)
/* Reset all event counters, not including PMCCNTR_EL0, to 0

 */
#define QUADD_ARMV8_PMCR_P      (1 << 1)
/* Reset PMCCNTR_EL0 to 0 */
#define QUADD_ARMV8_PMCR_C      (1 << 2)
/* Clock divider: PMCCNTR_EL0 counts every clock cycle/every 64 clock cycles */
#define QUADD_ARMV8_PMCR_D      (1 << 3)
/* Export of events is disabled/enabled */
#define QUADD_ARMV8_PMCR_X      (1 << 4)
/* Disable cycle counter, PMCCNTR_EL0 when event counting is prohibited */
#define QUADD_ARMV8_PMCR_DP     (1 << 5)
/* Long cycle count enable */
#define QUADD_ARMV8_PMCR_LC     (1 << 6)

static inline unsigned int armv8_pmu_pmcr_read(void)
{

	unsigned int val;
	/* Read Performance Monitors Control Register */
	asm volatile("mrs %0, pmcr_el0" : "=r" (val));
	return val;
}
static inline void armv8_pmu_pmcr_write(unsigned int val)
{
	asm volatile("msr pmcr_el0, %0" : :"r" (val & QUADD_ARMV8_PMCR_WR_MASK));
}

static void enable_all_counters(void)
{
	unsigned int val;
	/* Enable all counters */
	val = armv8_pmu_pmcr_read();
	val |= QUADD_ARMV8_PMCR_E | QUADD_ARMV8_PMCR_X;
	armv8_pmu_pmcr_write(val);
}

static void reset_all_counters(void)
{

	unsigned int val;
	val = armv8_pmu_pmcr_read();
	val |= QUADD_ARMV8_PMCR_P | QUADD_ARMV8_PMCR_C;
	armv8_pmu_pmcr_write(val);
}

static void readticks(unsigned int *result)
{
	struct timeval t;
	unsigned int cc;
	unsigned int val;
	if (!enabled) {
		reset_all_counters();
		enable_all_counters();
		enabled = 1;
	}
	cc = armv8_pmu_pmcr_read();
	gettimeofday(&t,(struct timezone *) 0);
	result[0] = cc;
	result[1] = t.tv_usec;
	result[2] = t.tv_sec;
}
