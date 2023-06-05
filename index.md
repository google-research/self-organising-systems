This page is intended as an introduction and ongoing status page for a research effort on Self Organising Systems. Self-organisation can be thought of as systems that consist of a large number of agents reaching global goals through local interactions. We focus on differentiable models and techniques to train them. Other terms that may describe this line of work include multi-agent cooperating models, but these tend to primarily be trained using reinforcement learning.

## Publications

{% assign sorted_pubs = site.data.publications | sort: 'published' | reverse %}
{% for publication in sorted_pubs %}
- [{{ publication.name }}]({{ publication.url }}) \| {{ publication.published | date: "%-d %B %Y" }} {% endfor %}

## Tutorials

{% assign sorted_tuts = site.data.tutorials | sort: 'published' | reverse %}
{% for tutorial in sorted_tuts %}
- [{{ tutorial.name }}]({{ tutorial.url }}) \| {{ tutorial.published | date: "%-d %B %Y" }} {% endfor %}
