

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 14})
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')



plt.figure(figsize=(6,4))
plt.plot([1, 2, 3, 4],'ks--', label='test', linewidth=2)
plt.legend()
plt.xlabel('N training samples')
plt.xticks([1, 2, 3, 4])
plt.ylabel('Time in s')
plt.xscale("log")
plt.yscale("log")
plt.savefig('figures/test.pdf', bbox_inches='tight')





print("Hello World du lappen")
