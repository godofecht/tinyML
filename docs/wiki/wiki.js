const navSections = document.getElementById('nav-sections');
const searchInput = document.getElementById('nav-search');
const docContent = document.getElementById('doc-content');
const docTitle = document.getElementById('doc-title');
const docSubtitle = document.querySelector('.doc-subtitle');
const breadcrumbs = document.getElementById('breadcrumbs');
const tocList = document.getElementById('toc-list');
const docCount = document.getElementById('doc-count');

const state = {
  sections: [],
  items: [],
  activeId: null,
};

const slugify = (text) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-');

const updateCount = (count) => {
  docCount.textContent = `${count} doc${count === 1 ? '' : 's'}`;
};

const buildNav = (filter = '') => {
  navSections.innerHTML = '';
  const filtered = filter.trim().toLowerCase();
  let visibleCount = 0;

  state.sections.forEach((section) => {
    const sectionEl = document.createElement('div');
    sectionEl.className = 'nav-section';

    const titleEl = document.createElement('div');
    titleEl.className = 'nav-section-title';
    titleEl.textContent = section.title;
    sectionEl.appendChild(titleEl);

    const listEl = document.createElement('ul');
    listEl.className = 'nav-list';

    section.items.forEach((item) => {
      if (filtered && !item.title.toLowerCase().includes(filtered)) {
        return;
      }
      const itemEl = document.createElement('li');
      itemEl.className = 'nav-item';
      if (item.id === state.activeId) {
        itemEl.classList.add('active');
      }

      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = item.title;
      button.addEventListener('click', () => loadDoc(item, true));
      itemEl.appendChild(button);
      listEl.appendChild(itemEl);
      visibleCount += 1;
    });

    if (listEl.children.length > 0) {
      sectionEl.appendChild(listEl);
      navSections.appendChild(sectionEl);
    }
  });

  updateCount(visibleCount);
};

const buildToc = () => {
  tocList.innerHTML = '';
  const headings = docContent.querySelectorAll('h2, h3');
  if (!headings.length) {
    tocList.innerHTML = '<div class="toc-empty">No sections</div>';
    return;
  }

  headings.forEach((heading) => {
    if (!heading.id) {
      heading.id = slugify(heading.textContent);
    }
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    if (heading.tagName === 'H3') {
      link.style.paddingLeft = '14px';
    }
    tocList.appendChild(link);
  });
};

const renderMarkdown = (markdown) => {
  if (window.marked) {
    marked.setOptions({
      gfm: true,
      mangle: false,
      headerIds: false,
    });
    docContent.innerHTML = marked.parse(markdown);
  } else {
    docContent.textContent = markdown;
  }

  const firstHeading = docContent.querySelector('h1');
  if (firstHeading) {
    docTitle.textContent = firstHeading.textContent;
    firstHeading.remove();
  }

  if (window.hljs) {
    window.hljs.highlightAll();
  }

  if (window.renderMathInElement) {
    window.renderMathInElement(docContent, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
    });
  }

  buildToc();
};

const loadDoc = async (item, pushState) => {
  try {
    const isHtml = item.path.toLowerCase().endsWith('.html');
    const response = await fetch(item.path);
    if (!response.ok) {
      throw new Error(`Failed to load ${item.path}`);
    }
    state.activeId = item.id;
    breadcrumbs.textContent = `TinyML / ${item.section} / ${item.title}`;
    docTitle.textContent = item.title;
    docSubtitle.textContent = item.section;
    if (isHtml) {
      docContent.innerHTML = `<iframe class=\"doc-frame\" src=\"${item.path}\" title=\"${item.title}\"></iframe>`;
      tocList.innerHTML = '<div class=\"toc-empty\">No sections</div>';
    } else {
      const markdown = await response.text();
      renderMarkdown(markdown);
    }
    buildNav(searchInput.value);
    if (pushState) {
      const url = new URL(window.location.href);
      url.searchParams.set('doc', item.id);
      history.replaceState({}, '', url);
    }
  } catch (error) {
    docContent.innerHTML = `<h2>Could not load document</h2><p>${error.message}</p>`;
  }
};

const init = async () => {
  const response = await fetch('content.json');
  const data = await response.json();

  state.sections = data.sections.map((section) => ({
    title: section.title,
    items: section.items.map((item, index) => ({
      ...item,
      section: section.title,
      id: `${slugify(section.title)}-${slugify(item.title)}-${index}`,
    })),
  }));

  state.items = state.sections.flatMap((section) => section.items);

  const params = new URLSearchParams(window.location.search);
  const requestedId = params.get('doc');
  const defaultItem = state.items.find((item) => item.id === requestedId) || state.items[0];

  buildNav();
  if (defaultItem) {
    loadDoc(defaultItem, false);
  }
};

searchInput.addEventListener('input', (event) => buildNav(event.target.value));

init();
