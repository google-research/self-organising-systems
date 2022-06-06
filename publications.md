## Publications

{% for publication in site.data.publications %}
- [{{ publication.name }}]({{ publication.url }})
{% endfor %}