import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probabilities = dict()
    transitions = corpus[page]

    if len(transitions) == 0:
        p_corpus = 1 / len(corpus)
        for p in corpus.keys():
            probabilities[p] = p_corpus
        
        return probabilities

    if len(transitions) > 0:
        p_corpus = (1 - damping_factor) / (len(transitions) + 1)
        p_page = (damping_factor / len(transitions)) + p_corpus

        probabilities[page] = p_corpus

        for p in transitions:
            probabilities[p] = p_page 
        
        return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    probabilities = dict()
    samples = []

    # Random first sample
    page = random.choice(list(corpus.keys()))
    samples.append(page)
    
    # Remaining samples after first
    for i in range(n-1):
        p = transition_model(corpus, page, damping_factor)
        page = random.choices(list(p.keys()), weights=list(p.values()), k=1)[0]
        samples.append(page)

    # Count
    for p in corpus.keys():
        probabilities[p] = samples.count(p) / n

    return probabilities


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # List all pages in corpus
    pages = list(corpus.keys())
    # {p: i}
    links = dict()

    # Fix corpus
    for p in corpus.keys():
        # If no links, then it has one link for every page in corpus
        if corpus[p] == set():
            corpus[p] = set(pages)
    
    for page in pages:
        links[page] = []
        for p in corpus.keys():
            if page in corpus[p]:
                links[page].append(p)
    #print(corpus)
    #print(links)

    probabilities = dict()
    updated_probabilities = dict()

    # Initial PR = 1/N
    for p in corpus.keys():
        probabilities[p] = 1 / len(corpus.keys())
        updated_probabilities[p] = float(0)

    # PR differences
    d = {k: abs(probabilities[k] - updated_probabilities[k]) for k in probabilities if k in updated_probabilities}

    # Recalculate
    i = 0
    p_corpus = (1 - damping_factor) / len(corpus)
    while max(d.values()) > 0.001:
        for p in corpus.keys():
            p_link = 0
            # Links
            for lp in links[p]:
                if (i % 2) == 0:
                    p_link += (probabilities[lp] / len(corpus[lp]))
                else:
                    p_link += (updated_probabilities[lp] / len(corpus[lp]))
            pr = p_corpus + (damping_factor * p_link)

            # Update probabilities or updated_probabilities dictionary
            if (i % 2) == 0:
                updated_probabilities[p] = pr
            else:
                probabilities[p] = pr
            
        # Increase count
        i += 1

        # Update differences dictionary
        d = {k: abs(probabilities[k] - updated_probabilities[k]) for k in probabilities if k in updated_probabilities}
        #print("P", "\033[93m {}\033[00m" .format(probabilities))
        #print("UP", "\033[96m {}\033[00m" .format(updated_probabilities))
        #print("D", "\033[91m {}\033[00m" .format(d))

    # When PR's do not change by > 0.001
    return probabilities


if __name__ == "__main__":
    main()
