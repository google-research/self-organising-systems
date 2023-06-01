This page is intended as an introduction and ongoing status page for a research effort on Self Organising Systems. Self-organisation can be thought of as systems that consist of a large number of agents reaching global goals through local interactions. We focus on differentiable models and techniques to train them. Other terms that may describe this line of work include multi-agent cooperating models, but these tend to primarily be trained using reinforcement learning.

## Publications

{% assign sorted_pubs = site.data.publications | sort: 'published' | reverse %}
{% for publication in sorted_pubs %}
- [{{ publication.name }}]({{ publication.url }}) \| {{ publication.published | date: "%-d %B %Y" }} {% endfor %}

## Tutorials

- 2022-09-23 [Simple 3D visualization with JAX raycasting](https://google-research.github.io/self-organising-systems/2022/jax-raycast/)
- 2022-06-06 [Differentiable Finite State Machines](https://google-research.github.io/self-organising-systems/2022/diff-fsm/)
